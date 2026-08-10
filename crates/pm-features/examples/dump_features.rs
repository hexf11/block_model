//! 用真实 ClickHouse 数据跑一遍特征计算，检查数值是否合理。
//!
//! 运行：
//!   cargo run -p pm-features --example dump_features
//!
//! 输出每个 symbol 在 T-110s 预测时点的完整特征向量，
//! 用于人工检查是否有 NAN 泛滥、量级异常等问题。

use pm_features::features::{compute_features, FEATURE_NAMES};
use pm_features::snapshot::SnapshotBuilder;

const CH_URL: &str = "http://127.0.0.1:8123/";
const SYMBOLS: [&str; 5] = ["btc/usd", "eth/usd", "sol/usd", "xrp/usd", "doge/usd"];

fn main() -> anyhow::Result<()> {
    let builder = SnapshotBuilder::new(CH_URL);

    // 找一个已完整采集的窗口：取 ClickHouse 里最新完整窗口的开始时间
    let now_ms = chrono::Utc::now().timestamp_millis();
    // 回退两个窗口，确保数据完整
    let win_start_ms = ((now_ms / 300_000) - 2) * 300_000;
    let snap_ms = win_start_ms + 190_000; // T-110s 预测时点

    println!("窗口开始: {win_start_ms}  预测时点: {snap_ms}");
    println!("（UTC {}）\n",
        chrono::DateTime::from_timestamp_millis(win_start_ms)
            .map(|d| d.format("%H:%M:%S").to_string())
            .unwrap_or_default());

    for symbol in SYMBOLS {
        let snap = builder.build(symbol, snap_ms)?;
        let fv = compute_features(&snap);

        println!("── {symbol} ──  ticks={}  open={:?}  current={:?}",
            snap.tick_count(), snap.open_price(), snap.current_price());

        let mut nan_count = 0;
        for (name, val) in FEATURE_NAMES.iter().zip(fv.features.iter()) {
            if val.is_nan() { nan_count += 1; }
            println!("  {name:<16} {val:>14.8}");
        }
        if nan_count > 0 {
            println!("  ⚠️  {nan_count} 个特征为 NAN");
        }
        println!();
    }

    Ok(())
}
