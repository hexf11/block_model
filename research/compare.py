#!/usr/bin/env python3
"""模型对比表 —— 读 models/*/metrics.json，按 test 集指标排名。

用法
----
    python research/compare.py
    python research/compare.py --models-dir models --sort brier

为什么以 Brier 为主
------------------
这个项目要的是「可信的 P(UP)」，不是「猜对方向」。Brier 同时惩罚
    - 区分不出方向（判别力差）
    - 概率没校准（说 80% 实际只有 55%）
准确率只看 0.5 阈值两侧，完全看不出校准好坏；AUC 只看排序，对概率
的绝对值免疫。所以主指标是 Brier，logloss 次之，ECE 单独看校准，
AUC 用来判断「是真没信号，还是只是没校准」。

只比校准后的数字
---------------
各脚本落进 metrics.json 的 train/val/test 指标都是 **套过各自
calibration.json 之后**的。原始输出的量纲各家不同（树模型是概率，
神经网络过了 softmax），不校准就比没有意义。

排名只看 test —— val 被校准器用掉了，train 会被过拟合污染。
"""

import argparse
import json
from pathlib import Path


def load_all(models_dir: Path) -> list[dict]:
    """扫 models/*/metrics.json。跳过没有 metrics.json 的子目录
    （比如只跑了一半的、或 shap/ 这种产物目录）。"""
    out = []
    for sub in sorted(models_dir.iterdir()):
        if not sub.is_dir():
            continue
        mp = sub / "metrics.json"
        if not mp.exists():
            continue
        try:
            m = json.loads(mp.read_text())
        except json.JSONDecodeError as e:
            print(f"跳过 {mp}：JSON 解析失败（{e}）")
            continue
        m["_dir"] = sub.name
        out.append(m)
    return out


def fmt(v, spec=".4f", na="  n/a "):
    return na if v is None else format(v, spec)


def consistency_warnings(models: list[dict]) -> list[str]:
    """对比前先确认这些模型确实可比 —— 同一份数据、同一个切分。

    不一致不一定是错（比如序列模型天然少 seq_len-1 条样本），但必须
    显式提示，否则排名会误导人。
    """
    warns = []
    if len(models) < 2:
        return warns

    ref = models[0]
    for m in models[1:]:
        if m.get("n_windows") != ref.get("n_windows"):
            warns.append(f"{m['_dir']} 的窗口数 {m.get('n_windows')} "
                         f"≠ {ref['_dir']} 的 {ref.get('n_windows')}")
        if m.get("time_range_ms") != ref.get("time_range_ms"):
            warns.append(f"{m['_dir']} 与 {ref['_dir']} 的时间范围不同 "
                         f"—— 不是同一份数据，排名无意义")
        if m.get("feature_cols") != ref.get("feature_cols"):
            warns.append(f"{m['_dir']} 与 {ref['_dir']} 的特征列不同")

    # 序列模型的评估样本数天然少于行数（滑窗吃掉开头 seq_len-1 条），
    # 这是预期内的差异，单独说明而不是当成错误。
    for m in models:
        te_n = (m.get("test") or {}).get("n")
        split_te = (m.get("split") or {}).get("test")
        if te_n is not None and split_te is not None and te_n != split_te:
            warns.append(f"{m['_dir']} 的 test 评估样本 {te_n} ≠ 行切分 {split_te}"
                         f"（seq_len={m.get('seq_len', '?')} 的滑窗损耗，属预期）")
    return warns


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default="models", help="模型产物根目录")
    ap.add_argument("--sort", default="brier", choices=["brier", "logloss", "auc", "ece"],
                    help="排名指标（test 集）。brier/logloss/ece 越小越好，auc 越大越好")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="用哪个切分排名（默认 test —— val 已被校准器用掉）")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        raise SystemExit(f"目录不存在: {models_dir}")

    models = load_all(models_dir)
    if not models:
        raise SystemExit(f"{models_dir}/ 下没有任何 metrics.json —— 先跑 train_*.py")

    for w in consistency_warnings(models):
        print(f"⚠️  {w}")
    print()

    # 排名：越小越好的指标升序，AUC 降序
    higher_better = args.sort == "auc"
    def key(m):
        v = (m.get(args.split) or {}).get(args.sort)
        if v is None:
            return float("-inf") if higher_better else float("inf")
        return -v if higher_better else v
    models.sort(key=key)

    hdr = (f"{'#':<3}{'模型':<14}{'n':>5}  {'Brier':>8}{'logloss':>10}"
           f"{'ECE':>9}{'AUC':>8}{'acc':>8}{'UP率':>8}")
    print(f"排名依据：{args.split} 集的 {args.sort}"
          f"（{'越大越好' if higher_better else '越小越好'}，均为校准后）")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for i, m in enumerate(models, 1):
        s = m.get(args.split) or {}
        print(f"{i:<3}{m['_dir']:<14}{s.get('n', 0):>5}  "
              f"{fmt(s.get('brier')):>8}{fmt(s.get('logloss')):>10}"
              f"{fmt(s.get('ece')):>9}{fmt(s.get('auc'), '.3f'):>8}"
              f"{fmt(s.get('acc'), '.3f'):>8}{fmt(s.get('up_rate'), '.3f'):>8}")
    print("=" * len(hdr))

    # 三个切分的 Brier 一起看 —— train ≫ test 说明过拟合
    print(f"\n各切分 Brier（train / val / test）:")
    for m in models:
        vals = " / ".join(fmt((m.get(k) or {}).get("brier")) for k in ("train", "val", "test"))
        print(f"  {m['_dir']:<14} {vals}")

    # 基线：常数预测 = 该切分的经验 UP 率。跑不赢它的模型没有价值。
    print(f"\n基线对照（{args.split} 集常数预测该集 UP 率）:")
    for m in models:
        s = m.get(args.split) or {}
        up = s.get("up_rate")
        if up is None:
            continue
        base = up * (1 - up)          # 常数预测 p=up_rate 的 Brier
        b = s.get("brier")
        verdict = "—" if b is None else ("✅ 优于基线" if b < base else "❌ 不如基线")
        print(f"  {m['_dir']:<14} 模型 {fmt(b)}  基线 {fmt(base)}  {verdict}")

    # ONNX 签名 —— 推理端据此自适应，换模型时要确认这里
    print("\nONNX 输出签名（推理端 crates/pm-inference 按此提取）:")
    for m in models:
        o = m.get("onnx") or {}
        if not o:
            print(f"  {m['_dir']:<14} （metrics.json 无 onnx 字段）")
            continue
        print(f"  {m['_dir']:<14} {o.get('output_name')} "
              f"({o.get('output_type')})  提取={o.get('extract')}  "
              f"偏差={o.get('max_abs_diff', float('nan')):.1e}")

    # 数据量提醒 —— 样本太少时上面的排名全是噪音
    n_win = models[0].get("n_windows")
    if n_win is not None and n_win < 1440:
        print(f"\n⚠️  只有 {n_win} 个窗口（约 {n_win / 5 / 12:.1f} 小时/币种）。"
              f"\n   样本太少，切分落在不同市场状态里，上面的排名基本是噪音。"
              f"\n   建议积累到 ≥1440 窗口（≈24h×5 币种）再下结论。")

    print(f"\n最优（{args.split}/{args.sort}）: {models[0]['_dir']}")


if __name__ == "__main__":
    main()
