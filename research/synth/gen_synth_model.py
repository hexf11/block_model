import json, numpy as np, lightgbm as lgb, onnxmltools, onnxruntime as ort
from skl2onnx.common.data_types import FloatTensorType
rng = np.random.default_rng(0)
N, F = 4000, 33
X = rng.normal(size=(N, F)).astype(np.float32)
y = ((X[:, 0] + 0.5*X[:, 1] + 1.5*rng.normal(size=N)) > 0).astype(int)
m = lgb.train({"objective": "binary", "num_leaves": 16, "learning_rate": 0.05,
               "min_data_in_leaf": 40, "feature_fraction": 0.7, "bagging_fraction": 0.8,
               "bagging_freq": 1, "lambda_l2": 2.0, "verbosity": -1},
              lgb.Dataset(X, label=y), num_boost_round=120)
onx = onnxmltools.convert_lightgbm(m, name="polymarket_updown",
      initial_types=[("input", FloatTensorType([None, 33]))], target_opset=15)
open("/tmp/synth_model.onnx", "wb").write(onx.SerializeToString())
sess = ort.InferenceSession("/tmp/synth_model.onnx", providers=["CPUExecutionProvider"])
out_names = [o.name for o in sess.get_outputs()]
def probs(arr_in):
    outs = sess.run(None, {"input": arr_in})
    for n, o in zip(out_names, outs):
        if n == "probabilities":
            return np.array([d[1] for d in o], dtype=np.float64)  # sequence of maps {0: P(down), 1: P(up)}
    raise RuntimeError("no probabilities output")
allp = probs(X)
q = np.quantile(allp, np.linspace(0, 1, 21))
print("x unique:", len(set(q.tolist())), "range:", round(float(q.min()),4), "~", round(float(q.max()),4))
cal = {"method": "isotonic_clip", "x": q.tolist(), "y": np.linspace(0.15, 0.85, 21).tolist(),
       "raw_min": float(q.min()), "raw_max": float(q.max()), "n": 21}
json.dump(cal, open("/tmp/calibration.json", "w"))
p5 = probs(X[:5])
open("/tmp/synth_X5.raw", "wb").write(X[:5].tobytes())
open("/tmp/synth_pup5.raw", "wb").write(p5.tobytes())
print("P(up)5:", np.round(p5, 8).tolist())
