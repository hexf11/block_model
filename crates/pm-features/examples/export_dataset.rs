//! 训练集导出：把 ClickHouse 里的窗口逐个还原成 T-110s 快照，
//! 计算特征超集，与标签一起写成 CSV。
//!
//! 运行：
//!   cargo run -p pm-features --release --example export_dataset -- data/train.csv
//!
//! 输出格式（每行一个训练样本）：
//!   symbol, win_start_ms, <FEATURE_COUNT 个特征>, label_up, rel_move, near_tie
//!
//! 安全保证：特征全部来自 WindowSnapshot（硬边界 obs_ms <= pred_ms），
//! 标签来自窗口结束后的 close_px。两者在时间上严格分离，不存在泄漏路径。
//!
//! 口径保证：同批窗口（同一 pred_ms）共享一次 USDT/USD 溢价估算，
//! 与推理路径 build_all() 完全一致 —— FX 敏感特征（spot_ret_from_open、
//! spot_norm_dist、twap_spot_gap_norm 等）训练时已剥离溢价，避免 train/serve skew。

use anyhow::{Context, Result};
use pm_features::features::{compute_features, FEATURE_NAMES};
use pm_features::snapshot::SnapshotBuilder;
use std::collections::BTreeMap;
use std::io::Write;

const CH_URL: &str = "http://127.0.0.1:8123/";

/// 一个窗口的标签信息（从 pm.training_windows 读取）
struct WindowLabel {
    symbol:       String,
    win_start_ms: i64,
    pred_ms:      i64,
    label_up:     u8,
    rel_move:     f64,
    near_tie:     u8,
}

/// 读取所有可训练窗口的标签。
///
/// 只取 quality='ok' 的窗口，且必须是已完整结束的窗口
/// （win_end_ms 已过去，否则 close_px 还不是最终值）。
fn fetch_labels() -> Result<Vec<WindowLabel>> {
    let now_ms = chrono::Utc::now().timestamp_millis();

    let sql = format!(
        "SELECT symbol, win_start_ms, pred_ms, label_up, rel_move, near_tie \
         FROM pm.training_windows \
         WHERE win_end_ms <= {now_ms} \
         ORDER BY win_start_ms ASC, symbol ASC \
         FORMAT JSONEachRow"
    );

    let url = format!("{CH_URL}?default_format=JSONEachRow");
    let resp = ureq::post(&url)
        .set("Content-Type", "text/plain; charset=utf-8")
        .send_bytes(sql.as_bytes())
        .context("查询 training_windows 失败")?;

    let text = resp.into_string()?;
    let mut out = Vec::new();

    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("JSON 解析失败: {line}"))?;

        // ClickHouse 的 UInt64 在 JSON 里是字符串，Float64 是数字
        let as_i64 = |k: &str| -> Result<i64> {
            match &v[k] {
                serde_json::Value::String(s) => s.parse::<i64>()
                    .with_context(|| format!("{k} 解析失败: {s}")),
                serde_json::Value::Number(n) => n.as_i64()
                    .with_context(|| format!("{k} 不是整数: {n}")),
                other => anyhow::bail!("{k} 格式意外: {other}"),
            }
        };
        let as_f64 = |k: &str| -> Result<f64> {
            match &v[k] {
                serde_json::Value::String(s) => s.parse::<f64>()
                    .with_context(|| format!("{k} 解析失败: {s}")),
                serde_json::Value::Number(n) => n.as_f64()
                    .with_context(|| format!("{k} 不是浮点: {n}")),
                other => anyhow::bail!("{k} 格式意外: {other}"),
            }
        };

        out.push(WindowLabel {
            symbol:       v["symbol"].as_str().context("symbol 缺失")?.to_string(),
            win_start_ms: as_i64("win_start_ms")?,
            pred_ms:      as_i64("pred_ms")?,
            label_up:     as_i64("label_up")? as u8,
            rel_move:     as_f64("rel_move")?,
            near_tie:     as_i64("near_tie")? as u8,
        });
    }

    Ok(out)
}

fn main() -> Result<()> {
    let out_path = std::env::args().nth(1)
        .unwrap_or_else(|| "data/train.csv".to_string());

    let labels = fetch_labels()?;
    if labels.is_empty() {
        anyhow::bail!("没有可导出的窗口 —— pm.training_windows 为空或全部尚未结束");
    }
    println!("从 ClickHouse 读到 {} 个已结束的可训练窗口", labels.len());

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut f = std::fs::File::create(&out_path)
        .with_context(|| format!("无法创建输出文件 {out_path}"))?;

    // CSV 表头
    write!(f, "symbol,win_start_ms")?;
    for name in FEATURE_NAMES { write!(f, ",{name}")?; }
    writeln!(f, ",label_up,rel_move,near_tie")?;

    let builder = SnapshotBuilder::new(CH_URL);
    let mut written  = 0usize;
    let mut skipped  = 0usize;
    let mut nan_rows = 0usize;

    // 按 pred_ms 分组：同一预测时点的所有 symbol 共享一次 USDT/USD 溢价估算。
    // 这一步是必须的 —— 推理路径走 build_all()，训练路径若逐个 build()
    // 就会少掉 FX 剥离，让 spot_* 系列特征在训练/推理之间口径不一致。
    let mut by_pred: BTreeMap<i64, Vec<&WindowLabel>> = BTreeMap::new();
    for lab in &labels {
        by_pred.entry(lab.pred_ms).or_default().push(lab);
    }
    println!("按预测时点分组：{} 个批次", by_pred.len());

    for (pred_ms, group) in &by_pred {
        // build_all 需要 &[&str]；同一批次内 symbol 唯一
        let syms: Vec<&str> = group.iter().map(|l| l.symbol.as_str()).collect();

        let snaps = match builder.build_all(&syms, *pred_ms) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  跳过整批 @ {pred_ms}: {e}");
                skipped += group.len();
                continue;
            }
        };

        for (lab, snap) in group.iter().zip(snaps.iter()) {
            // 快照必须落在正确的窗口上，否则说明时间对齐出了问题
            if snap.win_start_ms != lab.win_start_ms {
                eprintln!("  跳过 {} @ {}: 快照窗口 {} 与标签窗口不一致",
                    lab.symbol, lab.win_start_ms, snap.win_start_ms);
                skipped += 1;
                continue;
            }

            let fv = compute_features(snap);
            if fv.features.iter().any(|v| v.is_nan()) { nan_rows += 1; }

            write!(f, "{},{}", lab.symbol, lab.win_start_ms)?;
            for v in &fv.features {
                if v.is_nan() { write!(f, ",")?; }           // 空字段 = pandas 的 NaN
                else          { write!(f, ",{v:.10}")?; }
            }
            writeln!(f, ",{},{:.10},{}", lab.label_up, lab.rel_move, lab.near_tie)?;
            written += 1;
        }
    }

    println!("已写入 {out_path}：{written} 行");
    if skipped  > 0 { println!("  跳过 {skipped} 行（快照构建失败或窗口错位）"); }
    if nan_rows > 0 { println!("  {nan_rows} 行含 NAN 特征（训练时按需过滤）"); }

    // 标签分布 —— 严重偏斜说明标签逻辑或采集有问题
    let ups   = labels.iter().filter(|l| l.label_up == 1).count();
    let ties  = labels.iter().filter(|l| l.near_tie == 1).count();
    println!("\n标签分布：UP {} ({:.1}%) / DOWN {} ({:.1}%)",
        ups, 100.0 * ups as f64 / labels.len() as f64,
        labels.len() - ups, 100.0 * (labels.len() - ups) as f64 / labels.len() as f64);
    println!("近平局窗口：{ties} ({:.1}%)",
        100.0 * ties as f64 / labels.len() as f64);

    Ok(())
}
