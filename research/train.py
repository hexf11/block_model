#!/usr/bin/env python3
"""Polymarket 5min UP/DOWN —— 模型训练、校准、评估、ONNX 导出。

数据入口
--------
从 `export_dataset.rs` 导出的 CSV 读取（Rust 计算特征 —— 这是防
train/serve skew 的硬约定，Python 永远不重算特征）。CSV 生成方式：

    cargo run -p pm-features --release --example export_dataset -- data/train.csv

每行 = 一个 5 分钟窗口在 T-110s 预测时点的特征 + 标签。FX 溢价已由
build_all() 注入剥离，spot_* 系列特征不含 USDT/USD 偏移。

流程
----
1. 读 CSV，过滤（近平局窗口、NAN 过多的行）
2. 按时间序切分 train/val/test（时间序列 —— 绝不随机打乱切分）
3. LightGBM 训练（early stopping on val），symbol_id 作为 categorical
4. Isotonic 校准：在 val 上拟合，永远不碰 train/test
5. 评估：Brier、log loss、校准曲线、ECE、per-symbol 报告
6. 导出：model.onnx（onnxmltools）+ calibration.json（线性插值映射）
   —— ONNX 是推理端的特征契约，calibration.json 是配套校准器
7. 可选：--shap 输出 SHAP 汇总；--tune 跑 Optuna 调参

用法
----
    python research/train.py --data data/train.csv --out models/
    python research/train.py --data data/train.csv --out models/ --shap --tune
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

RNG_SEED = 42

# 诊断 / 导出用的特征子集
META_COLS = ["symbol", "win_start_ms", "label_up", "rel_move", "near_tie"]

# 缺失容忍：一行特征 NAN 数超过该值才丢弃。
# LightGBM 原生吃 NAN，少量缺失不必过滤，靠 min_data_in_leaf 兜底。
MAX_NAN_COLS = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="export_dataset.rs 输出的 CSV 路径")
    p.add_argument("--out", default="models", help="产物目录（model.onnx / calibration.json / metrics.json）")
    p.add_argument("--keep-near-tie", action="store_true",
                   help="默认丢弃 near_tie 窗口（|rel_move| < 2e-5，标签是噪音）")
    p.add_argument("--max-nan-cols", type=int, default=MAX_NAN_COLS,
                   help="特征 NAN 数超过该值的行被丢弃")
    p.add_argument("--shap", action="store_true", help="训练后跑 SHAP 分析")
    p.add_argument("--tune", action="store_true", help="先跑 Optuna 调参（慢）")
    p.add_argument("--trials", type=int, default=30, help="Optuna trials 数")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    return p.parse_args()


# ── 数据加载与过滤 ──────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        sys.exit("CSV 为空 —— 先跑 export_dataset.rs")

    missing = [c for c in META_COLS if c not in df.columns]
    if missing:
        sys.exit(f"CSV 缺少元信息列 {missing} —— export_dataset.rs 输出格式不对")

    feat_cols = [c for c in df.columns if c not in META_COLS]
    print(f"读入 {len(df)} 行，{len(feat_cols)} 个特征列")
    print(f"特征: {feat_cols}")

    # win_start_ms 是 i64，pandas 读成 int64 没问题；symbol 是 str
    df = df.assign(
        win_start_ms=df["win_start_ms"].astype(np.int64),
        symbol=df["symbol"].astype(str),
        label_up=df["label_up"].astype(np.int8),
    )

    # 标签完整性
    if df["label_up"].isna().any():
        sys.exit("label_up 存在空值 —— 导出有问题")

    # symbol_id 是 categorical（整数编码）。LightGBM 的 categorical 必须是
    # 整数类型 —— export_dataset.rs 输出的是 "0.0" 这种 float 文本，必须转 int
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
    绝不随机打乱 —— 相邻窗口高度相关，随机切分会高估泛化能力。
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


def make_lgb_datasets(train, val, feat_cols, cat_cols):
    """构造 LightGBM Dataset，把 categorical 声明放在 Dataset 上。

    LightGBM 的 categorical_feature 接受列名或整数位置。列名在版本间
    有解析差异（4.5.0 不认 "name:" 前缀），整数位置最稳 ——
    feat_cols 是导出契约里固定顺序的特征名列表，位置映射唯一。
    """
    import lightgbm as lgb

    X_tr = train[feat_cols]
    X_va = val[feat_cols]

    cat_idx = [feat_cols.index(c) for c in cat_cols] if cat_cols else None

    dtr = lgb.Dataset(X_tr, label=train["label_up"], categorical_feature=cat_idx)
    dva = lgb.Dataset(X_va, label=val["label_up"], reference=dtr)
    return dtr, dva


# ── 校准 ───────────────────────────────────────────────────────────────────

def calibrate_isotonic(model, val, feat_cols, out_path: Path):
    """在 val 上拟合 isotonic 校准器（训练/校准集严格分离）。

    校准器序列化为 calibration.json —— 推理端 Rust 用线性插值复现，
    与 sklearn IsotonicRegression(out_of_bounds='clip') 语义一致。
    """
    p_val = model.predict(val[feat_cols])
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(p_val, val["label_up"])

    # 退化情况：val 上模型输出为常数（如数据量太少时），isotonic 只剩一个
    # 点，无法插值。降级为常数映射：校准后概率 = val 上的经验 UP 率。
    # 推理端 isotonic_clip() 对 n==1 的表返回 y[0]，两侧语义一致。
    if len(iso.X_thresholds_) < 2:
        emp = float(np.mean(val["label_up"]))
        calib = {
            "method": "isotonic_clip",
            "x": [float(p_val[0])],
            "y": [emp],
            "raw_min": float(p_val.min()),
            "raw_max": float(p_val.max()),
            "n": int(len(p_val)),
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
    }
    out_path.write_text(json.dumps(calib, indent=2))
    print(f"校准器已写 {out_path}  （{calib['n']} 个 val 样本）")
    return calib


def apply_calibration(p: np.ndarray, calib: dict) -> np.ndarray:
    return np.interp(p, calib["x"], calib["y"])


# ── 评估 ───────────────────────────────────────────────────────────────────

def evaluate(p: np.ndarray, y: np.ndarray, name: str) -> dict:
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
        # 单一类别时 Brier 仍有定义，但要显式给 pos_label 避免歧义
        brier = float(np.mean((g_p - y) ** 2))
        lines.append(f"{sym:<8} {len(grp):5d}  {y.mean():.3f}   "
                     f"{g_p.mean():.3f}    {brier:.4f}")
    return "\n".join(lines)


# ── 导出 ───────────────────────────────────────────────────────────────────

def export_onnx(model, feat_cols, out_path: Path) -> None:
    """LightGBM Booster → ONNX。ONNX 的输入名 = 推理端契约。

    输出节点带 sigmoid，即为原始 P(UP)；推理端再套 calibration.json
    的线性插值得到最终概率。
    """
    from onnxmltools import convert_lightgbm
    from skl2onnx.common.data_types import FloatTensorType

    initial_types = [("input", FloatTensorType([None, len(feat_cols)]))]
    onx = convert_lightgbm(model, name="polymarket_updown", initial_types=initial_types,
                           target_opset=15)  # skl2onnx converter 支持的上限
    out_path.write_bytes(onx.SerializeToString())
    print(f"ONNX 已写 {out_path}  （{len(feat_cols)} 个输入特征，opset 15）")


def shap_analysis(model, df, feat_cols, out_dir: Path) -> None:
    import shap

    X = df[feat_cols]
    # TreeExplainer 对 LightGBM Booster 原生支持
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 特征重要性 CSV（mean|SHAP|）—— 无 matplotlib 也能产出
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

    # 汇总图（需要 matplotlib，缺了就跳过，不影响 CSV）
    try:
        import matplotlib.pyplot as plt
        shap.summary_plot(sv, X, feature_names=feat_cols, show=False, max_display=33)
        # shap>=0.46 的 summary_plot 不再返回 figure，从 pyplot 取当前图
        fig = plt.gcf()
        fig_path = out_dir / "shap_summary.png"
        fig.savefig(fig_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"SHAP 汇总图已写 {fig_path}")
    except ImportError:
        print("未安装 matplotlib，跳过 SHAP 汇总图（CSV 已产出）")


def optuna_tune(train, val, feat_cols, cat_cols, n_trials: int):
    import lightgbm as lgb
    import optuna

    # 每个 trial 现造 Dataset —— min_data_in_leaf 会在 trial 间变化，
    # LightGBM 不允许在 pre-filtered 的 Dataset 上动态调小它
    def make_datasets():
        X_tr = train[feat_cols]
        X_va = val[feat_cols]
        cat_idx = [feat_cols.index(c) for c in cat_cols] if cat_cols else None
        dtr = lgb.Dataset(X_tr, label=train["label_up"], categorical_feature=cat_idx)
        dva = lgb.Dataset(X_va, label=val["label_up"], reference=dtr)
        return dtr, dva

    def objective(trial):
        params = dict(
            objective="binary",
            metric="binary_logloss",
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 63),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 30, 300),
            feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            bagging_freq=1,
            lambda_l2=trial.suggest_float("lambda_l2", 0.0, 5.0),
            verbosity=-1,
            seed=RNG_SEED,
            feature_pre_filter=False,   # 允许 trial 间改 min_data_in_leaf
        )
        dtr, dva = make_datasets()
        m = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        return m.best_score["valid_0"]["binary_logloss"]

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=RNG_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"Optuna 完成：best logloss = {study.best_value:.5f}")
    print(f"最佳参数: {study.best_params}")
    return study.best_params


# ── main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    df = filter_rows(df, args.keep_near_tie, args.max_nan_cols)

    feat_cols = [c for c in df.columns if c not in META_COLS]
    cat_cols = ["symbol_id"] if "symbol_id" in feat_cols else []

    train, val, test = chronological_split(df, args.val_frac, args.test_frac)

    # ── 可选：Optuna 调参 ──
    best_params = None
    if args.tune:
        best_params = optuna_tune(train, val, feat_cols, cat_cols, args.trials)

    import lightgbm as lgb
    params = dict(
        objective="binary",
        metric="binary_logloss",
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=100,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        verbosity=-1,
        seed=RNG_SEED,
    )
    if best_params:
        params.update(best_params)

    dtr, dva = make_lgb_datasets(train, val, feat_cols, cat_cols)
    model = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False),
                                 lgb.log_evaluation(100)])
    print(f"最佳迭代: {model.best_iteration}  最佳 val logloss: {model.best_score['valid_0']['binary_logloss']:.5f}")

    # 模型文本（可读诊断）与 ONNX
    model.save_model(str(out_dir / "model.txt"))
    export_onnx(model, feat_cols, out_dir / "model.onnx")

    # ── 校准（只在 val 上拟合）──
    calib = calibrate_isotonic(model, val, feat_cols, out_dir / "calibration.json")

    # ── 评估 ──
    metrics = {"feature_cols": feat_cols}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        p_raw = model.predict(part[feat_cols])
        p = apply_calibration(p_raw, calib)
        m = evaluate(p, part["label_up"].values, name)
        metrics[name] = m

    # 校准曲线（仅 test —— train/val 的校准曲线会因过拟合校准器而过于好看）
    print("\n校准曲线（test，按预测分桶）:")
    print(reliability_table(apply_calibration(model.predict(test[feat_cols]), calib),
                            test["label_up"].values))
    print("\nPer-symbol（test）:")
    p_test = apply_calibration(model.predict(test[feat_cols]), calib)
    print(per_symbol_report(test, p_test))

    # 元信息
    metrics["symbols"] = sorted(df["symbol"].unique().tolist())
    metrics["n_windows"] = int(len(df))
    metrics["time_range_ms"] = [int(df["win_start_ms"].min()), int(df["win_start_ms"].max())]
    metrics["calibration"] = calib
    metrics["best_iteration"] = int(model.best_iteration)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\n指标已写 {metrics_path}")

    # ── SHAP ──
    if args.shap:
        shap_analysis(model, pd.concat([train, val, test]), feat_cols, out_dir / "shap")


if __name__ == "__main__":
    main()
