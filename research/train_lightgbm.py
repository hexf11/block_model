#!/usr/bin/env python3
"""LightGBM 训练 —— Polymarket 5min UP/DOWN。

数据入口
--------
从 `export_dataset.rs` 导出的 CSV 读取（Rust 计算特征 —— 这是防
train/serve skew 的硬约定，Python 永远不重算特征）。CSV 生成方式：

    cargo run -p pm-features --release --example export_dataset -- data/train.csv

共享的数据加载 / 时间序切分 / 校准 / 评估都在 common.py，本文件只负责
「LightGBM 怎么训」。产物写 models/lightgbm/。

用法
----
    python research/train_lightgbm.py
    python research/train_lightgbm.py --shap --tune
"""

import argparse
from pathlib import Path

import lightgbm as lgb

import common
from common import RNG_SEED

MODEL_NAME = "lightgbm"

BASE_PARAMS = dict(
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


def make_datasets(train, val, feat_cols, cat_cols):
    """构造 LightGBM Dataset，把 categorical 声明放在 Dataset 上。

    LightGBM 的 categorical_feature 接受列名或整数位置。列名在版本间
    有解析差异（4.5.0 不认 "name:" 前缀），整数位置最稳 ——
    feat_cols 是导出契约里固定顺序的特征名列表，位置映射唯一。
    """
    cat_idx = [feat_cols.index(c) for c in cat_cols] if cat_cols else None
    dtr = lgb.Dataset(train[feat_cols], label=train["label_up"],
                      categorical_feature=cat_idx)
    dva = lgb.Dataset(val[feat_cols], label=val["label_up"], reference=dtr)
    return dtr, dva


def tune(train, val, feat_cols, cat_cols, n_trials: int) -> dict:
    import optuna

    def objective(trial):
        params = dict(
            BASE_PARAMS,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 63),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 30, 300),
            feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            lambda_l2=trial.suggest_float("lambda_l2", 0.0, 5.0),
            feature_pre_filter=False,   # 允许 trial 间改 min_data_in_leaf
        )
        # 每个 trial 现造 Dataset —— LightGBM 不允许在 pre-filtered 的
        # Dataset 上动态调小 min_data_in_leaf
        dtr, dva = make_datasets(train, val, feat_cols, cat_cols)
        m = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        return m.best_score["valid_0"]["binary_logloss"]

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=RNG_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"Optuna 完成：best logloss = {study.best_value:.5f}")
    print(f"最佳参数: {study.best_params}")
    return study.best_params


def export_onnx(model, feat_cols, out_path: Path):
    """LightGBM Booster → ONNX。onnxmltools 走 ZipMap，
    输出是 sequence<map<int64,float>>，每行 {0: P(down), 1: P(up)}。"""
    from onnxmltools import convert_lightgbm
    from skl2onnx.common.data_types import FloatTensorType

    initial_types = [("input", FloatTensorType([None, len(feat_cols)]))]
    onx = convert_lightgbm(model, name="polymarket_updown",
                           initial_types=initial_types,
                           target_opset=15)  # skl2onnx converter 支持的上限
    common.export_onnx_bytes(onx, out_path, len(feat_cols))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(p)
    args = p.parse_args()

    out_dir = Path(args.out or f"models/{MODEL_NAME}")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = common.prepare(args)
    train, val, test = data["train"], data["val"], data["test"]
    feat_cols, cat_cols = data["feat_cols"], data["cat_cols"]

    params = dict(BASE_PARAMS)
    if args.tune:
        params.update(tune(train, val, feat_cols, cat_cols, args.trials))

    dtr, dva = make_datasets(train, val, feat_cols, cat_cols)
    model = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False),
                                 lgb.log_evaluation(100)])
    print(f"最佳迭代: {model.best_iteration}  "
          f"最佳 val logloss: {model.best_score['valid_0']['binary_logloss']:.5f}")

    model.save_model(str(out_dir / "model.txt"))   # 可读诊断
    export_onnx(model, feat_cols, out_dir / "model.onnx")

    def predict(X):
        return model.predict(X)

    # 校准只在 val 上拟合，绝不碰 train/test
    calib = common.calibrate_isotonic(predict(val[feat_cols]), val["label_up"].values,
                                      out_dir / "calibration.json")

    # ONNX 签名实测：喂 test（含 NAN）验证与 Booster 一致
    sig = common.verify_onnx(out_dir / "model.onnx", test[feat_cols], predict(test[feat_cols]))

    common.report_and_save(MODEL_NAME, predict, data, calib, out_dir,
                           extra={"onnx": sig,
                                  "params": params,
                                  "best_iteration": int(model.best_iteration)})

    if args.shap:
        import pandas as pd
        common.shap_analysis(model, pd.concat([train, val, test]),
                             feat_cols, out_dir / "shap")


if __name__ == "__main__":
    main()
