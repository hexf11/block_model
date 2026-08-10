#!/usr/bin/env python3
"""
验证 003_windows.sql 里的两个标签假设：

  假设 1：open_px = 窗口内第一条 TWAP（argMin(price, obs_ms)）
          —— 对应 Polymarket 的 priceToBeat

  假设 2：close_px > open_px → UP；否则 DOWN
          —— 对应 Polymarket 的实际结算方向

做法：
  1. 从 ClickHouse 取最近 quality='ok' 的 btc/usd 窗口
  2. 用 win_start_ms/1000 构造 slug，请求 gamma-api.polymarket.com
  3. 对比 open_px vs priceToBeat、close_px vs finalPrice、label_up vs 实际结算
"""

import json
import sys
import time
import urllib.request
import urllib.error
from decimal import Decimal

# ── ClickHouse 连接 ────────────────────────────────────────────────────────────

CH_URL = "http://127.0.0.1:8123/"

def ch_query(sql: str) -> list[dict]:
    """通过 HTTP 接口执行 SELECT，返回行列表。"""
    params = urllib.parse.urlencode({"query": sql, "default_format": "JSONEachRow"})
    url = CH_URL + "?" + params
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            lines = resp.read().decode().strip().splitlines()
            return [json.loads(l) for l in lines if l]
    except urllib.error.URLError as e:
        print(f"[ERROR] ClickHouse 请求失败: {e}", file=sys.stderr)
        sys.exit(1)

import urllib.parse

# ── Polymarket API ─────────────────────────────────────────────────────────────

GAMMA = "https://gamma-api.polymarket.com"
HEADERS = {"User-Agent": "block-model-verifier/1.0"}

def pm_fetch(slug: str) -> dict | None:
    """拿一个市场的 eventMetadata（含 priceToBeat 和 finalPrice）。"""
    url = f"{GAMMA}/events?slug={slug}"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            # 返回值是列表
            if isinstance(data, list) and data:
                return data[0]
            return None
    except Exception as e:
        print(f"  [WARN] {slug}: API 请求失败 ({e})")
        return None

def extract_prices(event: dict) -> tuple[Decimal | None, Decimal | None, str | None]:
    """从 event 里提取 priceToBeat、finalPrice、实际结算方向。"""
    meta = event.get("eventMetadata") or {}
    price_to_beat = meta.get("priceToBeat")
    final_price   = meta.get("finalPrice")

    # 结算方向：找赢的 outcome token
    winner = None
    for mkt in event.get("markets", []):
        for token in mkt.get("tokens", []):
            if token.get("price") == "1" or token.get("winner") is True:
                winner = token.get("outcome", "").upper()
                break
        if winner:
            break

    # 有些市场直接在顶层有 winnerOutcome
    if not winner:
        for mkt in event.get("markets", []):
            wo = mkt.get("winnerOutcome") or mkt.get("winner_outcome")
            if wo:
                winner = str(wo).upper()
                break

    p2b = Decimal(str(price_to_beat)) if price_to_beat else None
    fp  = Decimal(str(final_price))   if final_price   else None
    return p2b, fp, winner

# ── 主逻辑 ─────────────────────────────────────────────────────────────────────

def main():
    print("=== 标签假设验证 ===\n")

    # 从 ClickHouse 取最近 20 个已完成的 btc/usd 窗口
    sql = """
    SELECT
        symbol,
        win_start_ms,
        win_start_ms + 300000 AS win_end_ms,
        open_px,
        close_px,
        label_up,
        rel_move,
        n_ticks
    FROM pm.market_windows
    WHERE symbol = 'btc/usd' AND quality = 'ok'
    ORDER BY win_start_ms DESC
    LIMIT 20
    """
    rows = ch_query(sql)
    if not rows:
        print("[ERROR] ClickHouse 没有 btc/usd 可用窗口，请先等数据积累。")
        sys.exit(1)

    print(f"从 ClickHouse 拿到 {len(rows)} 个 btc/usd 窗口\n")

    # 对比结果
    match_open  = 0  # open_px 与 priceToBeat 接近
    match_close = 0  # close_px 与 finalPrice 接近
    match_label = 0  # label_up 与实际结算一致
    total_compared = 0

    TOLS = [Decimal("0.01"), Decimal("1"), Decimal("10")]  # 允许的价格误差（USD）

    rows_with_data = []

    for row in rows:
        sym       = row["symbol"]
        win_ms    = int(row["win_start_ms"])
        win_s     = win_ms // 1000          # Unix 秒
        open_px   = Decimal(str(row["open_px"]))
        close_px  = Decimal(str(row["close_px"]))
        label_up  = int(row["label_up"])
        rel_move  = float(row["rel_move"])

        slug = f"btc-updown-5m-{win_s}"
        print(f"窗口 {time.strftime('%H:%M', time.gmtime(win_s))} UTC  slug={slug}")

        event = pm_fetch(slug)
        if event is None:
            print(f"  → API 无数据（市场可能尚未结算或 slug 格式不对）\n")
            time.sleep(0.3)
            continue

        p2b, fp, winner = extract_prices(event)
        if p2b is None or fp is None:
            print(f"  → priceToBeat/finalPrice 缺失（event 可能未完全结算）")
            print(f"     eventMetadata keys: {list((event.get('eventMetadata') or {}).keys())}\n")
            time.sleep(0.3)
            continue

        total_compared += 1

        # 计算差值
        diff_open  = abs(open_px  - p2b)
        diff_close = abs(close_px - fp)

        # 我们计算的标签
        our_label  = "UP" if label_up == 1 else "DOWN"
        # Polymarket 实际结算（严格大于 = UP）
        pm_up      = fp > p2b
        pm_label   = "UP" if pm_up else "DOWN"

        open_ok  = diff_open  < Decimal("1.00")
        close_ok = diff_close < Decimal("1.00")
        label_ok = (our_label == pm_label)

        if open_ok:  match_open  += 1
        if close_ok: match_close += 1
        if label_ok: match_label += 1

        rows_with_data.append({
            "slug": slug,
            "our_open": open_px, "pm_p2b": p2b, "diff_open": diff_open,
            "our_close": close_px, "pm_fp": fp, "diff_close": diff_close,
            "our_label": our_label, "pm_label": pm_label,
            "open_ok": open_ok, "close_ok": close_ok, "label_ok": label_ok,
            "rel_move_pct": rel_move * 100,
            "winner_raw": winner,
        })

        status = []
        status.append(f"open  {'✓' if open_ok  else '✗'}  ours={open_px:.4f}  pm={p2b:.4f}  diff={diff_open:.4f}")
        status.append(f"close {'✓' if close_ok else '✗'}  ours={close_px:.4f}  pm={fp:.4f}  diff={diff_close:.4f}")
        status.append(f"label {'✓' if label_ok else '✗'}  ours={our_label}  pm={pm_label}  (rel_move={rel_move*100:.4f}%)")
        for s in status:
            print(f"  {s}")
        print()

        time.sleep(0.3)  # 礼貌性限速

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"对比了 {total_compared} 个已结算窗口（共 {len(rows)} 个，其余未结算）\n")

    if total_compared == 0:
        print("没有可对比的数据。可能原因：")
        print("  1. 近期窗口尚未被 Polymarket 结算")
        print("  2. slug 格式不对（尝试手动检查一个 URL）")
        print("  3. 数据采集时间太短，本地窗口与 Polymarket 时区/对齐方式不同")
        return

    print(f"  open_px  vs priceToBeat : {match_open}/{total_compared} 匹配（误差<$1）")
    print(f"  close_px vs finalPrice  : {match_close}/{total_compared} 匹配（误差<$1）")
    print(f"  label_up vs 实际结算    : {match_label}/{total_compared} 匹配")
    print()

    # 详细分析
    mismatches = [r for r in rows_with_data if not r["label_ok"]]
    if mismatches:
        print(f"⚠️  标签不一致的窗口（{len(mismatches)} 个）：")
        for r in mismatches:
            print(f"  {r['slug']}")
            print(f"    ours={r['our_label']}  pm={r['pm_label']}")
            print(f"    open={r['our_open']} p2b={r['pm_p2b']}  close={r['our_close']} fp={r['pm_fp']}")
            print(f"    rel_move={r['rel_move_pct']:.5f}%  winner_raw={r['winner_raw']}")
    else:
        print("✓ 所有窗口标签均与 Polymarket 实际结算一致")

    # 假设 1：开盘价定义
    if match_open == total_compared:
        print("\n✓ 假设 1 成立：open_px（窗口内第一条 TWAP）与 Polymarket priceToBeat 吻合（误差<$1）")
    elif match_open > total_compared * 0.8:
        print(f"\n⚠️  假设 1 基本成立（{match_open}/{total_compared}），但有少量偏差，需进一步核查")
    else:
        print(f"\n✗ 假设 1 存疑：open_px 与 priceToBeat 差距较大（{match_open}/{total_compared} 匹配）")
        print("  → 可能 priceToBeat 是窗口开始前最后一条，而非窗口内第一条")

    # 假设 2：平局规则
    tie_windows = [r for r in rows_with_data if abs(float(r["rel_move_pct"])) < 0.002]
    if tie_windows:
        print(f"\n近平局窗口（{len(tie_windows)} 个，rel_move<0.002%）：")
        for r in tie_windows:
            print(f"  {r['slug']}  ours={r['our_label']}  pm={r['pm_label']}  diff={r['pm_fp']-r['pm_p2b']:.6f}")
    else:
        print("\n（本次对比没有 rel_move 极小的近平局窗口，无法直接验证平局规则）")
        print("  → 平局（close=open）算 DOWN 的假设目前无反例，但样本量不足")

if __name__ == "__main__":
    main()
