#!/usr/bin/env python3
"""TCN-LSTM 训练脚本 —— PyTorch 版，第二套模型，与 LightGBM 共享 common.py。

数据从哪来 / 序列怎么组装
------------------------
不重写 Rust 导出器。复用 data/train.csv（export_dataset.rs 已按 pred_ms
滑窗输出，每行是一个窗口的一条快照，特征顺序与 LightGBM 完全一致），
在这里用 PyTorch 的 Dataset 做滑窗：

    每条样本(i) = 固定长度 seq_len 的时间序列，取 i 及其之前 seq_len-1 个
    相邻窗口。每条样本共用最后一行的 label_up（预测的就是该窗口的涨跌）。

因为是「同一份 CSV、同一个 fold、同一套指标」，和 LightGBM 的对比才公平。
序列只是把同一份 X 重排成 [N, seq_len, F]。

时间序切分怎么做（关键）
----------------------
common.py 的 chronological_split 先切行（train/val/test 各落一个时段），
再用这些行区间在重排后的序列上切。
    - 序列 i 覆盖行 (i-seq_len+1) .. i，标签取自最后一行 i。
    - 一段区间 [an, bn] 内构造出的序列号是 [an+seq_len-1, bn]。
    - 用「序列最后一行所属切分」来归属：给定该切分的行号集合 S，
      它覆盖的序列号 = [min(S)+seq_len-1, max(S)]。三个切分各自取闭区间，
      相邻段首尾恰好相接、无重叠也无缺口 —— 与 LightGBM 的逐行分组完全一致。
网络在构造序列时会越过 train 段看到 val 段的开头几个窗口（滑窗需要往前
看 seq_len-1 个），这是模型的合法输入（推理时服务端缓存的就是这些窗口），
不是泄漏 —— 只有标签不跨界，而标签永远来自序列最后一行。

NAN 怎么处理（与树模型的本质差异）
--------------------------------
dense 网络吃不了 NAN。数值特征置零填充，并用 concat 上 isnan 掩码通道
（[B,T,2F_nan_able]）让网络能「知道」哪些值在填充。symbol_id 走 categorical
embedding。树模型原生吃 NAN，语义不同 —— 这是光照/不光照之外的第二处差异，
正是我们要对比的东西。

与 common.py 的分工
------------------
    data/filter/split/calibration/eval/verify_onnx   → common.py
    模型定义、序列组装、训练循环、ONNX 导出           → 本文件
产物写 models/tcn_lstm/。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    RNG_SEED, add_common_args, prepare, calibrate_isotonic,
    apply_calibration, evaluate, reliability_table, per_symbol_report,
)

torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)


# ── 序列组装 ──────────────────────────────────────────────────────────────

def build_sequences(df: pd.DataFrame, feat_cols: list[str], seq_len: int):
    """把同一份 X 重排成序列。X 按 win_start_ms 全局排序（收集器已按
    (symbol, win_start_ms) 聚簇，所有符号步长相同 → 单一时间轴）。

    返回 (X_vals, X_isnan, ids, y, last_row_idx)：
      X_vals    [N, seq_len, n_val_feats]   数值特征，NAN 已置 0
      X_isnan   [N, seq_len, n_val_feats]   同位置 NAN 掩码（1=缺失）
      ids       [N, seq_len]                symbol_id（embedding 用）
      y         [N]                         label_up（取序列最后一行）
      last_row_idx [N]                      每条序列对应 CSV 的最后一行 index
    """
    df = df.sort_values("win_start_ms").reset_index(drop=True)
    val_feats = [c for c in feat_cols if c != "symbol_id"]
    n_val = len(val_feats)

    raw = np.asarray(df[val_feats].values, dtype=np.float64)
    Xv = np.nan_to_num(raw).astype(np.float32)          # NAN → 0
    Xi = np.isnan(raw).astype(np.float32)               # 1=缺失，正常=0

    ids = df["symbol_id"].round().astype(np.int64).values

    n = len(df)
    N = n - seq_len + 1
    Xs = np.zeros((N, seq_len, n_val), dtype=np.float32)
    Is = np.zeros((N, seq_len), dtype=np.int64)
    Ns = np.zeros((N, seq_len, n_val), dtype=np.float32)
    ys = np.zeros(N, dtype=np.float32)
    last = np.arange(seq_len - 1, n, dtype=np.int64)

    for j, i in enumerate(last):
        Xs[j] = Xv[i - seq_len + 1:i + 1]
        Is[j] = ids[i - seq_len + 1:i + 1]
        Ns[j] = Xi[i - seq_len + 1:i + 1]
        ys[j] = df["label_up"].values[i]

    if len(Xs) == 0:
        sys.exit(f"seq_len={seq_len} ≥ 样本数，构造不了任何序列 —— 调小 --seq-len")
    return Xs, Ns, Is, ys, last


# ── 模型 ────────────────────────────────────────────────────────────────────

class TCNLSTM(nn.Module):
    """TCN + 双向 LSTM 级联。

    输入 [B, T, F_val] + [B, T] symbol_id。
    数值特征与 isnan 掩码 concat 后过 TCN（因果空洞卷积）；symbol 走 embedding
    后过 TCN 的通道 concat。结果过 BiLSTM，全局平均池化，MLP 到 logits [B,2]。
    返回 logits —— 集成时套 sigmoid / softmax 才得到概率。
    """

    def __init__(self, n_val_feats, n_symbols, seq_len,
                 n_kernels=32, kernel=3, lstm_hidden=32, n_layers=1,
                 embed_dim=8, hidden=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(n_symbols, embed_dim)
        # TCN 输入通道 = 数值特征 + isnan 掩码 + symbol embedding
        self.n_time = n_val_feats + n_val_feats + embed_dim

        tcn = []
        in_ch = self.n_time
        for ch, dil in [(n_kernels, 1), (n_kernels, 2)]:
            tcn.append(nn.Conv1d(in_ch, ch, kernel,
                                 padding=(kernel - 1) * dil, dilation=dil))
            tcn.append(nn.ReLU())
            tcn.append(nn.BatchNorm1d(ch))
            tcn.append(nn.Dropout(dropout))
            in_ch = ch
        self.tcn = nn.Sequential(*tcn)

        self.lstm = nn.LSTM(n_kernels, lstm_hidden, n_layers,
                            batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, mask, ids):
        emb = self.embed(ids)                          # [B,T,embed]
        x = torch.cat([x, mask], dim=-1)               # [B,T,2F_val]
        x = torch.cat([x, emb], dim=-1)                # [B,T,n_time]
        t = self.tcn(x.transpose(1, 2))                # [B,ch,T]
        t = t.transpose(1, 2)                          # [B,T,ch]
        out, _ = self.lstm(t)                          # [B,T,2*lstm_hidden]
        pooled = out.mean(dim=1)                       # global avg pool
        return self.head(pooled)                       # [B,2] logits


# ── 训练 ─────────────────────────────────────────────────────────────────

class SeqData(torch.utils.data.Dataset):
    def __init__(self, Xs, Ns, Is, ys):
        self.Xs, self.Ns, self.Is, self.ys = Xs, Ns, Is, ys
    def __len__(self):
        return len(self.ys)
    def __getitem__(self, i):
        return (torch.from_numpy(self.Xs[i]), torch.from_numpy(self.Ns[i]),
                torch.from_numpy(self.Is[i]), self.ys[i])


def model_predict(model, ds, device):
    """对整段序列数据集批量运行，返回未校准的 P(UP)。"""
    model.eval()
    ps = []
    with torch.no_grad():
        for xb, nb_, ib, _ in torch.utils.data.DataLoader(ds, batch_size=512):
            xb, nb_, ib = xb.to(device), nb_.to(device), ib.to(device)
            logits = model(xb, nb_, ib)
            ps.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(ps)


def logits_to_pup(logits):
    return torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()


def fit(model, train_dl, val_ds, n_epochs, lr, device):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    best_val = None
    for ep in range(n_epochs):
        model.train()
        tot, nb = 0.0, 0
        for xb, nb_, ib, yb in train_dl:
            xb, nb_, ib, yb = (xb.to(device), nb_.to(device),
                               ib.to(device), yb.to(device))
            opt.zero_grad()
            logits = model(xb, nb_, ib)
            loss = lossf(logits[:, 1], yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
            nb += len(yb)
        # 每轮在 val 上算 Brier，早停
        vp = model_predict(model, val_ds, device)
        vb = float(np.mean((vp - val_ds.ys) ** 2))
        if (ep + 1) % max(1, n_epochs // 5) == 0:
            print(f"  epoch {ep+1:3d}  train_loss {tot/nb:.4f}  val_brier {vb:.4f}")
        if best_val is None or vb < best_val:
            best_val, best_state = vb, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best_state


def _split_sequences(last: np.ndarray, train: pd.DataFrame,
                     val: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """把序列号 {序列 i 的最后一行 = last[i]} 划进 train/val/test。

    序列可能跨切分边界（向前看 seq_len-1 个窗口），但标签只来自最后一行，
    所以归属只看 last[i]。train/val/test 段互不重叠、首尾相接。
    """
    def seg(part):
        rows = set(part.index.tolist())
        return np.array([int(r) in rows for r in last.tolist()])
    return seg(train), seg(val), seg(test)


class ExportWrapper(nn.Module):
    """导出专用包装：把训练时的 logits 输出改成概率。

    训练用 BCEWithLogitsLoss（数值稳定，需要 logits），但推理端 Rust 的
    predict() 有 [0,1] 断言 —— 导出 logits 会被直接拒掉。这里在图里补上
    softmax，让 ONNX 的输出就是 [P(down), P(up)]，与 model_predict()
    的 torch.softmax(...)[:,1] 完全同一语义。
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, mask, ids):
        return torch.softmax(self.model(x, mask, ids), dim=-1)


def verify_onnx_seq(onnx_path: Path, ds, p_ref: np.ndarray,
                    tol: float = 1e-5) -> dict:
    """跑 onnxruntime，确认 ONNX 输出 == PyTorch 的 P(UP)，并记录签名。

    common.verify_onnx 假设单个 DataFrame 输入，序列模型是三路输入，
    所以单写一份。作用一样：不实测就不知道推理端该怎么取输出。
    """
    import onnxruntime as rt

    sess = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    feed = {in_names[0]: ds.Xs.astype(np.float32),
            in_names[1]: ds.Ns.astype(np.float32),
            in_names[2]: ds.Is.astype(np.int64)}
    res = sess.run(None, feed)

    out_meta = sess.get_outputs()[0]
    arr = np.asarray(res[0])
    assert arr.ndim == 2 and arr.shape[1] == 2, f"期望 [N,2] 概率张量，实际 {arr.shape}"
    p = arr[:, 1].astype(np.float64)
    diff = float(np.max(np.abs(p - np.asarray(p_ref, dtype=np.float64))))
    if diff > tol:
        sys.exit(f"ONNX 校验失败：与 PyTorch 最大偏差 {diff:.2e} > {tol}")

    # 概率必须落在 [0,1]，否则 Rust 端 predict() 的断言会拒
    assert float(arr.min()) >= -1e-6 and float(arr.max()) <= 1 + 1e-6, \
        f"ONNX 输出不在 [0,1]：min={arr.min()} max={arr.max()} —— 漏了 softmax"

    sig = {
        "input_names": in_names,
        "input_shapes": [[d if isinstance(d, int) else str(d) for d in i.shape]
                         for i in sess.get_inputs()],
        "output_name": out_meta.name,
        "output_type": out_meta.type,
        "extract": "tensor",
        "max_abs_diff": diff,
    }
    print(f"ONNX 校验通过：输出 '{out_meta.name}' ({out_meta.type})，"
          f"提取方式=tensor，与 PyTorch 最大偏差 {diff:.2e}")
    return sig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--seq-len", type=int, default=16, help="滑窗序列长度")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n-kernels", type=int, default=32)
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--lstm-hidden", type=int, default=32)
    ap.add_argument("--embed-dim", type=int, default=8)
    ap.add_argument("--n-epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = prepare(args)
    df, feat_cols = data["df"], data["feat_cols"]
    train, val, test = data["train"], data["val"], data["test"]

    Xs, Ns, Is, ys, last = build_sequences(df, feat_cols, args.seq_len)
    tr_m, va_m, te_m = _split_sequences(last, train, val, test)

    n_symbols = int(df["symbol_id"].nunique())
    model = TCNLSTM(Xs.shape[-1], n_symbols, args.seq_len,
                    n_kernels=args.n_kernels, kernel=args.kernel,
                    lstm_hidden=args.lstm_hidden, n_layers=1,
                    embed_dim=args.embed_dim, hidden=args.hidden,
                    dropout=args.dropout).to(device)

    tr_dl = torch.utils.data.DataLoader(
        SeqData(Xs[tr_m], Ns[tr_m], Is[tr_m], ys[tr_m]),
        batch_size=args.batch, shuffle=True)
    val_ds = SeqData(Xs[va_m], Ns[va_m], Is[va_m], ys[va_m])
    fit(model, tr_dl, val_ds, args.n_epochs, args.lr, device)

    out_dir = Path(args.out or "models/tcn_lstm")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 校准（val 上拟合，与 LightGBM 完全同一语义）
    val_p = model_predict(model, val_ds, device)
    calib = calibrate_isotonic(val_p, val_ds.ys, out_dir / "calibration.json")

    # ONNX 导出（plain tensor；导出图内套 softmax，输出直接是概率）
    export_onnx(model, out_dir / "model.onnx", Xs.shape[-1], args.seq_len, device)

    # 各切分的未校准 P(UP) —— 序列与行一一对应，直接对齐
    seg_data = {
        "train": SeqData(Xs[tr_m], Ns[tr_m], Is[tr_m], ys[tr_m]),
        "val": SeqData(Xs[va_m], Ns[va_m], Is[va_m], ys[va_m]),
        "test": SeqData(Xs[te_m], Ns[te_m], Is[te_m], ys[te_m]),
    }
    def predict_set(name):
        ds = seg_data[name]
        return model_predict(model, ds, device)

    # ONNX 签名实测：用 test 序列验证 ONNX 与 PyTorch 一致
    sig = verify_onnx_seq(out_dir / "model.onnx", seg_data["test"],
                          predict_set("test"))

    metrics = {"model": "tcn_lstm", "feature_cols": feat_cols}
    for name in ["train", "val", "test"]:
        p = apply_calibration(predict_set(name), calib)
        metrics[name] = evaluate(np.asarray(p), seg_data[name].ys, name)

    # 校准曲线 / per-symbol 只用 test（与 LightGBM 口径一致）
    p_test = apply_calibration(predict_set("test"), calib)
    print("\n校准曲线（test，按预测分桶）:")
    print(reliability_table(p_test, seg_data["test"].ys))
    print("\nPer-symbol（test）:")
    print(per_symbol_report(test, p_test))

    metrics["symbols"] = sorted(df["symbol_id"].unique().tolist())
    metrics["n_windows"] = int(len(df))
    metrics["time_range_ms"] = [int(df["win_start_ms"].min()), int(df["win_start_ms"].max())]
    metrics["split"] = {"train": len(train), "val": len(val), "test": len(test)}
    metrics["sequence_split"] = {"train": int(tr_m.sum()), "val": int(va_m.sum()),
                                 "test": int(te_m.sum())}
    metrics["calibration"] = calib
    metrics["seq_len"] = args.seq_len
    metrics["architecture"] = "tcn_lstm"
    metrics["onnx"] = sig
    metrics["params"] = {"n_kernels": args.n_kernels, "kernel": args.kernel,
                         "lstm_hidden": args.lstm_hidden, "hidden": args.hidden,
                         "embed_dim": args.embed_dim, "dropout": args.dropout,
                         "lr": args.lr, "n_epochs": args.n_epochs,
                         "batch": args.batch}

    json_path = out_dir / "metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\n指标已写 {json_path}")
    return metrics


def export_onnx(model, onnx_path: Path, n_val_feats: int, seq_len: int,
                device, dummy_seed: int = 0):
    """导出 plain-tensor ONNX。输出 [B,2] 概率 —— 推理端 Rust 按
    OutputKind::Tensor2 取第 1 列。三路输入：数值、isnan 掩码、symbol id。

    为什么用 dynamo=False（legacy TorchScript 导出器）
    ------------------------------------------------
    torch 2.9+ 默认的 torch.export 导出器对 **双向 LSTM** 会生成错误的
    Reshape：BiLSTM 的输出是 [B,T,2,hidden]，图里却硬 reshape 成
    [B,1,2*hidden]，onnxruntime 跑起来直接报
        input_shape_size == requested_shape_size was false
    legacy 导出器对同一个模型是正确的（实测与 PyTorch 输出 max diff = 0）。
    等上游修好 LSTM 的 dynamo 路径再切回默认导出器。
    """
    model.eval()
    wrapped = ExportWrapper(model).to(device)
    wrapped.eval()
    dummy_x = torch.zeros(1, seq_len, n_val_feats, device=device) + dummy_seed
    dummy_n = torch.zeros(1, seq_len, n_val_feats, device=device)
    dummy_i = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped, (dummy_x, dummy_n, dummy_i), str(onnx_path),
        input_names=["seq_val", "seq_mask", "seq_id"],
        output_names=["probabilities"],
        dynamo=False,
        dynamic_axes={"seq_val": {0: "batch"}, "seq_mask": {0: "batch"},
                      "seq_id": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version=15)
    print(f"ONNX 已写 {onnx_path}")


if __name__ == "__main__":
    main()
