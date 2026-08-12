#!/usr/bin/env python3
"""多模型训练的共享层 —— 数据、切分、校准、评估、导出。

为什么要有这个文件
------------------
模型之间的对比只有在「同一份数据、同一个 fold、同一套指标」下才有意义。
把这些抽到 common.py，每个 train_*.py 就只剩「怎么训练这个模型」，
不可能各自偷偷改切分比例或评估口径。

各脚本的分工：

    common.py          数据加载 / 时间序切分 / isotonic 校准 / 评估 / ONNX 校验
    train_lightgbm.py  ─┐
    train_xgboost.py    ├─ 只写模型定义与拟合，产物写 models/{name}/
    train_catboost.py  ─┘
    compare.py         读 models/*/metrics.json 出对比表

ONNX 输出签名
-------------
不同转换器的输出结构不一样：onnxmltools 的树模型走 ZipMap，导出
`sequence<map<int64,float>>`；别的可能是普通 tensor。verify_onnx() 会
实测签名并写进 metrics.json，推理端（crates/pm-inference）据此自适应。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score

RNG_SEED = 42

# 非特征列 —— 元信息与标签，不进模型
META_COLS = ["symbol", "win_start_ms", "label_up", "rel_move", "near_tie", "pred_ms"]

# 缺失容忍：一行特征 NAN 数超过该值才丢弃。
# 树模型原生吃 NAN，少量缺失不必过滤。
MAX_NAN_COLS = 5


# ── CLI ────────────────────────────────────────────────────────────────────

def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """所有 train_*.py 共享的参数。切分比例故意放这里 ——
    各脚本不应该有自己的默认值，否则对比就不公平了。"""
    p.add_argument("--data", default="data/train.csv",
                   help="export_dataset.rs 输出的 CSV 路径")
    p.add_argument("--out", default=None,
                   help="产物目录，默认 models/{模型名}/")
    p.add_argument("--keep-near-tie", action="store_true",
                   help="默认丢弃 near_tie 窗口（|rel_move| < 2e-5，标签是噪音）")
    p.add_argument("--max-nan-cols", type=int, default=MAX_NAN_COLS,
                   help="特征 NAN 数超过该值的行被丢弃")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--tune", action="store_true", help="先跑 Optuna 调参（慢）")
    p.add_argument("--trials", type=int, default=30, help="Optuna trials 数")
    p.add_argument("--shap", action="store_true", help="训练后跑 SHAP 分析")
    return p


# ── 数据加载与过滤 ──────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        sys.exit("CSV 为空 —— 先跑 export_dataset.rs")

    # pred_ms 是后加的列，老的 CSV 可能没有 —— 只强制核心元信息
    required = [c for c in META_COLS if c != "pred_ms"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"CSV 缺少元信息列 {missing} —— export_dataset.rs 输出格式不对")

    feat_cols = [c for c in df.columns if c not in META_COLS]
    print(f"读入 {len(df)} 行，{len(feat_cols)} 个特征列")

    df = df.assign(
        win_start_ms=df["win_start_ms"].astype(np.int64),
        symbol=df["symbol"].astype(str),
        label_up=df["label_up"].astype(np.int8),
    )

    if df["label_up"].isna().any():
        sys.exit("label_up 存在空值 —— 导出有问题")

    # symbol_id 是 categorical（整数编码）。export_dataset.rs 输出的是
    # "0.0" 这种 float 文本，各家 categorical API 都要整数类型。
    if "symbol_id" in df.columns:
        df = df.assign(symbol_id=df["symbol_id"].astype(np.int32))

    return df


def filter_rows(df: pd.DataFrame, keep_near_tie: bool, max_nan_cols: int) -> pd.DataFrame:
    feat_cols = [c for c in df.columns if c not in META_COLS]
    start = len(df)

    if not keep_near_tie:
        n_before = len(df)
        df = df[df["near_tie"] != 1]
        print(f"丢弃 near_tie 窗口 {n_before - len(df)} 行")

    n_before = len(df)
    n_nan = df[feat_cols].isna().sum(axis=1)
    df = df[n_nan <= max_nan_cols]
    print(f"丢弃特征缺失过多的行 {n_before - len(df)} 行（阈值 >{max_nan_cols} 个 NAN）")

    print(f"过滤后剩 {len(df)} 行（共丢 {start - len(df)} 行）")
    if len(df) == 0:
        sys.exit("过滤后没有样本了 —— 检查 near_tie / NAN 阈值")
    return df.reset_index(drop=True)


def chronological_split(df: pd.DataFrame, val_frac: float, test_frac: float):
    """时间序切分：train 最早，val 次之，test 最新。

    绝不随机打乱 —— 相邻窗口高度相关，随机切分会严重高估泛化能力。
    所有模型必须共用这个函数，否则对比无效。
    """
    df = df.sort_values("win_start_ms").reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_test - n_val

    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]

    print(f"切分（时间序）: train={len(train)}  val={len(val)}  test={len(test)}")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        rng = (part["win_start_ms"].min(), part["win_start_ms"].max())
        print(f"  {name:<5} {part['label_up'].mean():.3f} UP  时间范围 {rng[0]}..{rng[1]}")
    return train, val, test


def prepare(args) -> dict:
    """一次性完成 加载 → 过滤 → 切分，返回所有脚本需要的东西。

    返回 dict 而不是 tuple —— 字段多了以后位置解包容易错位。
    """
    df = load_data(args.data)
    df = filter_rows(df, args.keep_near_tie, args.max_nan_cols)

    feat_cols = [c for c in df.columns if c not in META_COLS]
    cat_cols = ["symbol_id"] if "symbol_id" in feat_cols else []

    train, val, test = chronological_split(df, args.val_frac, args.test_frac)
    return dict(df=df, feat_cols=feat_cols, cat_cols=cat_cols,
                train=train, val=val, test=test)


# ── 校准 ───────────────────────────────────────────────────────────────────

def calibrate_isotonic(p_val: np.ndarray, y_val: np.ndarray, out_path: Path) -> dict:
    """在 val 上拟合 isotonic 校准器（训练/校准集严格分离）。

    接收概率数组而不是 model —— 各家 predict API 不一样，调用方负责取
    P(UP)，这里只管校准，对所有模型语义一致。

    校准器序列化为 calibration.json，推理端 Rust 用线性插值复现，
    与 sklearn IsotonicRegression(out_of_bounds='clip') 语义一致。
    """
    p_val = np.asarray(p_val, dtype=np.float64)
    y_val = np.asarray(y_val)

    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(p_val, y_val)

    # 退化情况：val 上模型输出为常数（数据量太少时常见），isotonic 只剩
    # 一个点，无法插值。降级为常数映射：校准后概率 = val 上的经验 UP 率。
    # 推理端 isotonic_clip() 对 n==1 的表返回 y[0]，两侧语义一致。
    if len(iso.X_thresholds_) < 2:
        emp = float(np.mean(y_val))
        calib = {
            "method": "isotonic_clip",
            "x": [float(p_val[0])],
            "y": [emp],
            "raw_min": float(p_val.min()),
            "raw_max": float(p_val.max()),
            "n": int(len(p_val)),
            "degenerate": True,
        }
        out_path.write_text(json.dumps(calib, indent=2))
        print(f"校准器已写 {out_path}  （退化：常数映射 → {emp:.4f}）")
        return calib

    calib = {
        "method": "isotonic_clip",
        "x": iso.X_thresholds_.tolist(),   # 原始模型输出 p_raw
        "y": iso.y_thresholds_.tolist(),   # 校准后概率
        # 训练期间 val 上的经验范围，供推理端 sanity check
        "raw_min": float(p_val.min()),
        "raw_max": float(p_val.max()),
        "n": int(len(p_val)),
        "degenerate": False,
    }
    out_path.write_text(json.dumps(calib, indent=2))
    print(f"校准器已写 {out_path}  （{calib['n']} 个 val 样本）")
    return calib


def apply_calibration(p: np.ndarray, calib: dict) -> np.ndarray:
    # 单点表：np.interp 对长度 1 的 xp 返回常数，与 Rust 的 n==1 分支一致
    return np.interp(p, calib["x"], calib["y"])


# ── 评估 ───────────────────────────────────────────────────────────────────

def evaluate(p: np.ndarray, y: np.ndarray, name: str) -> dict:
    """校准后概率的评估。Brier 是主指标 —— 它同时惩罚
    「区分不出方向」和「概率没校准」，正是这个项目要的。"""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y)
    # 小样本切分可能只含单一类别 —— AUC 无定义，logloss 需显式给 labels
    single_class = len(np.unique(y)) < 2
    m = dict(
        name=name,
        n=int(len(y)),
        up_rate=float(y.mean()),
        brier=float(np.mean((p - y) ** 2)),
        logloss=float(log_loss(y, p, labels=[0, 1])),
        acc=float(((p >= 0.5).astype(int) == y).mean()),
        auc=None if single_class else float(roc_auc_score(y, p)),
        ece=compute_ece(p, y),
    )
    auc_s = "n/a " if m["auc"] is None else f"{m['auc']:.3f}"
    print(f"[{name}] n={m['n']}  UP率={m['up_rate']:.3f}  "
          f"Brier={m['brier']:.4f}  logloss={m['logloss']:.4f}  "
          f"acc={m['acc']:.3f}  AUC={auc_s}  ECE={m['ece']:.4f}")
    return m


def _bin_mask(p: np.ndarray, lo: float, hi: float, is_last: bool) -> np.ndarray:
    """分桶掩码。最后一桶右闭，否则 p == 1.0 会落在所有桶之外。"""
    return (p >= lo) & (p <= hi) if is_last else (p >= lo) & (p < hi)


def compute_ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error —— 校准质量的主要指标。"""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(p)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = _bin_mask(p, lo, hi, i == n_bins - 1)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def reliability_table(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> str:
    """校准曲线表：pred 分桶 vs 实际 UP 率。"""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = ["bin        n     pred      actual   |diff|"]
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = _bin_mask(p, lo, hi, i == n_bins - 1)
        n = int(mask.sum())
        if n == 0:
            rows.append(f"{lo:.1f}-{hi:.1f}   {n:5d}   (empty)")
            continue
        pred, act = p[mask].mean(), y[mask].mean()
        rows.append(f"{lo:.1f}-{hi:.1f}   {n:5d}   {pred:.3f}     {act:.3f}    {abs(pred-act):.3f}")
    return "\n".join(rows)


def per_symbol_report(test: pd.DataFrame, p: np.ndarray) -> str:
    """每个 symbol 的 UP 率 / 预测均值 / Brier —— 供信心阈值与分层用。

    p 与 test 按行位置对齐（不是按 index）—— chronological_split 后
    test 的 index 不从 0 起，用 .loc 取会错位。
    """
    lines = ["symbol   n    up_rate  pred_mean  brier"]
    pos = {ix: i for i, ix in enumerate(test.index)}
    for sym, grp in test.groupby("symbol", sort=False):
        g_p = p[[pos[ix] for ix in grp.index]]
        y = grp["label_up"].values
        brier = float(np.mean((g_p - y) ** 2))
        lines.append(f"{sym:<8} {len(grp):5d}  {y.mean():.3f}   "
                     f"{g_p.mean():.3f}    {brier:.4f}")
    return "\n".join(lines)


# ── ONNX 校验 ──────────────────────────────────────────────────────────────

def _extract_pup(out):
    """从一个 ONNX 输出里尝试取 P(UP)，取不出返回 None。

    覆盖三种签名：
      - ZipMap（onnxmltools/skl2onnx 树模型）: [{0: p_down, 1: p_up}, ...]
      - 二分类概率张量 [N, 2]: 取第 1 列
      - 单列概率张量 [N, 1] / [N]: 直接用（需是浮点，排除 label 输出）
    """
    if isinstance(out, list):
        if out and isinstance(out[0], dict):
            key = 1 if 1 in out[0] else (max(out[0]) if out[0] else None)
            if key is None:
                return None
            return np.array([d[key] for d in out], dtype=np.float64)
        return None
    arr = np.asarray(out)
    if not np.issubdtype(arr.dtype, np.floating):
        return None      # label 输出是整数，不是概率
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 1].astype(np.float64)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0].astype(np.float64)
    if arr.ndim == 1:
        return arr.astype(np.float64)
    return None


def verify_onnx(onnx_path: Path, X: pd.DataFrame, p_ref: np.ndarray,
                tol: float = 1e-5) -> dict:
    """跑 onnxruntime，确认 ONNX 输出 == 训练框架的预测，并记录输出签名。

    这一步是多模型框架的安全网：每个转换器的输出结构都不一样，
    不实测就不知道推理端（Rust）该走哪条提取路径。签名写进 metrics.json。

    注意 X 里保留 NAN —— 树模型把缺失当独立分支，ONNX 侧必须复现同样
    的语义，喂 NAN 才能验出来。
    """
    import onnxruntime as rt

    sess = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    in_shape = sess.get_inputs()[0].shape
    res = sess.run(None, {in_name: X.to_numpy(dtype=np.float32)})

    p_ref = np.asarray(p_ref, dtype=np.float64)
    matched = None
    for out_meta, out in zip(sess.get_outputs(), res):
        p = _extract_pup(out)
        if p is None or len(p) != len(p_ref):
            continue
        diff = float(np.max(np.abs(p - p_ref)))
        if diff <= tol:
            matched = (out_meta, out, diff)
            break

    if matched is None:
        names = [(o.name, o.type) for o in sess.get_outputs()]
        sys.exit(f"ONNX 校验失败：没有任何输出能复现训练框架的 P(UP)。"
                 f"输出节点={names}  参考值前 3 个={p_ref[:3]}")

    out_meta, out, diff = matched
    kind = "zipmap" if isinstance(out, list) else "tensor"
    sig = {
        "input_name": in_name,
        "input_shape": [d if isinstance(d, int) else str(d) for d in in_shape],
        "output_name": out_meta.name,
        "output_type": out_meta.type,
        "extract": kind,
        "max_abs_diff": diff,
    }
    print(f"ONNX 校验通过：输出 '{out_meta.name}' ({out_meta.type})，"
          f"提取方式={kind}，与训练框架最大偏差 {diff:.2e}")
    return sig


def export_onnx_bytes(onx, out_path: Path, n_feats: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(onx.SerializeToString())
    print(f"ONNX 已写 {out_path}  （{n_feats} 个输入特征）")


# ── 收尾：评估 + metrics.json ───────────────────────────────────────────────

def report_and_save(model_name: str, predict_fn, data: dict, calib: dict,
                    out_dir: Path, extra: dict | None = None) -> dict:
    """三个切分上评估、打校准曲线与 per-symbol，落 metrics.json。

    predict_fn(X: DataFrame) -> np.ndarray 返回未校准的 P(UP)。
    各模型的 predict API 不同，由调用方包一层。
    """
    df, feat_cols = data["df"], data["feat_cols"]
    train, val, test = data["train"], data["val"], data["test"]

    metrics = {"model": model_name, "feature_cols": feat_cols}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        p = apply_calibration(predict_fn(part[feat_cols]), calib)
        metrics[name] = evaluate(p, part["label_up"].values, name)

    # 校准曲线只看 test —— train/val 的曲线会因校准器在 val 上拟合而过于好看
    p_test = apply_calibration(predict_fn(test[feat_cols]), calib)
    print("\n校准曲线（test，按预测分桶）:")
    print(reliability_table(p_test, test["label_up"].values))
    print("\nPer-symbol（test）:")
    print(per_symbol_report(test, p_test))

    metrics["symbols"] = sorted(df["symbol"].unique().tolist())
    metrics["n_windows"] = int(len(df))
    metrics["time_range_ms"] = [int(df["win_start_ms"].min()), int(df["win_start_ms"].max())]
    metrics["split"] = {"train": len(train), "val": len(val), "test": len(test)}
    metrics["calibration"] = calib
    if extra:
        metrics.update(extra)

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\n指标已写 {metrics_path}")
    return metrics


def shap_analysis(explainer_model, df, feat_cols, out_dir: Path) -> None:
    """SHAP 特征重要性。TreeExplainer 支持 LightGBM/XGBoost/CatBoost。"""
    import shap

    X = df[feat_cols]
    explainer = shap.TreeExplainer(explainer_model)
    sv = explainer.shap_values(X)
    # 有的版本对二分类返回 [class0, class1] 两份，取正类
    if isinstance(sv, list):
        sv = sv[-1]

    out_dir.mkdir(parents=True, exist_ok=True)
    imp = pd.DataFrame({
        "feature": feat_cols,
        "mean_abs_shap": np.abs(sv).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    csv_path = out_dir / "shap_importance.csv"
    imp.to_csv(csv_path, index=False)
    print(f"SHAP 重要性已写 {csv_path}")
    print("\nTop 10 特征：")
    for _, row in imp.head(10).iterrows():
        print(f"  {row['feature']:<20} {row['mean_abs_shap']:.5f}")

    try:
        import matplotlib.pyplot as plt
        shap.summary_plot(sv, X, feature_names=feat_cols, show=False,
                          max_display=len(feat_cols))
        # shap>=0.46 的 summary_plot 不再返回 figure，从 pyplot 取当前图
        fig = plt.gcf()
        fig_path = out_dir / "shap_summary.png"
        fig.savefig(fig_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"SHAP 汇总图已写 {fig_path}")
    except ImportError:
        print("未安装 matplotlib，跳过 SHAP 汇总图（CSV 已产出）")
