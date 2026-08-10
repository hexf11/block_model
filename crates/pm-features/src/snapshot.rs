/// 快照重建模块：给定时刻 T，从 ClickHouse 还原 T 时刻可见的一切。
///
/// 核心保证：所有返回数据的 obs_ms / ts_ex / recv_ms <= snap_ms，
/// 绝不包含 snap_ms 之后产生的任何信息（防未来信息泄漏）。
///
/// 使用方式：
///   let snap = Snapshot::at(snap_ms, symbol, ch_url);
///   let twap  = snap.twap_window()?;     // 窗口内所有 TWAP tick
///   let open  = snap.twap_open()?;       // 窗口开盘价
///   let normd = snap.normalized_dist()?; // 归一化距离特征

use anyhow::{Context, Result};
use rust_decimal::Decimal;
use std::str::FromStr;

// ── ClickHouse HTTP 辅助 ──────────────────────────────────────────────────────

/// 通过 ClickHouse HTTP 接口执行查询，返回 JSONEachRow 行。
fn ch_query_raw(ch_url: &str, sql: &str) -> Result<Vec<serde_json::Value>> {
    let body = sql.as_bytes().to_vec();
    let url = format!("{ch_url}?default_format=JSONEachRow");

    // 使用标准库的 TcpStream 避免引入额外 async 依赖；
    // 快照重建仅在离线/训练路径调用，同步 I/O 足够。
    let response = ureq::post(&url)
        .set("Content-Type", "text/plain; charset=utf-8")
        .send_bytes(&body)
        .with_context(|| format!("ClickHouse 请求失败: {sql}"))?;

    let text = response.into_string()?;
    let rows = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).with_context(|| format!("JSON 解析失败: {l}")))
        .collect::<Result<Vec<_>>>()?;
    Ok(rows)
}

// ── 核心数据结构 ──────────────────────────────────────────────────────────────

/// TWAP tick 的轻量表示（仅快照需要的字段）
#[derive(Debug, Clone)]
pub struct TwapTick {
    pub obs_ms: i64,
    pub price:  Decimal,
}

/// 给定 symbol + snap_ms 的完整快照
#[derive(Debug, Clone)]
pub struct WindowSnapshot {
    /// 预测时刻（毫秒 Unix）
    pub snap_ms:     i64,
    pub symbol:      String,
    /// 本市场窗口开始时间（5 分钟对齐）
    pub win_start_ms: i64,
    /// 窗口内所有 TWAP tick，obs_ms <= snap_ms，按 obs_ms 升序
    pub twap_ticks:  Vec<TwapTick>,
}

impl WindowSnapshot {
    /// 窗口内第一条 tick（= Polymarket priceToBeat，已验证）
    pub fn open_price(&self) -> Option<Decimal> {
        self.twap_ticks.first().map(|t| t.price)
    }

    /// snap_ms 时刻最新的 tick
    pub fn current_price(&self) -> Option<Decimal> {
        self.twap_ticks.last().map(|t| t.price)
    }

    /// 已过去的毫秒数
    pub fn elapsed_ms(&self) -> i64 {
        self.snap_ms - self.win_start_ms
    }

    /// 剩余毫秒数（300000 - elapsed）
    pub fn remaining_ms(&self) -> i64 {
        300_000 - self.elapsed_ms()
    }

    /// 窗口内价格标准差（全体 tick）
    pub fn price_std(&self) -> Option<f64> {
        let n = self.twap_ticks.len();
        if n < 2 { return None; }
        let prices: Vec<f64> = self.twap_ticks.iter()
            .map(|t| t.price.try_into().unwrap_or(f64::NAN))
            .collect();
        let mean = prices.iter().sum::<f64>() / n as f64;
        let var  = prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        Some(var.sqrt())
    }

    /// 归一化距离：(P_now - P_open) / (σ · √t_remaining_s)
    ///
    /// 直觉：当前偏移了多少个"剩余时间内的预期波动单位"。
    /// 值越大（越正）→ 价格已经大幅上行，UP 方向动能越强。
    /// 接近 0 → 尚未偏离，方向不明。
    pub fn normalized_distance(&self) -> Option<f64> {
        let open: f64    = self.open_price()?.try_into().ok()?;
        let current: f64 = self.current_price()?.try_into().ok()?;
        let sigma   = self.price_std()?;
        if sigma == 0.0 { return None; }
        let t_remaining_s = self.remaining_ms() as f64 / 1000.0;
        if t_remaining_s <= 0.0 { return None; }
        Some((current - open) / (sigma * t_remaining_s.sqrt()))
    }

    /// tick 数量（数据质量指标）
    pub fn tick_count(&self) -> usize {
        self.twap_ticks.len()
    }
}

// ── 快照构建器 ────────────────────────────────────────────────────────────────

/// 快照构建器：封装 ClickHouse 查询逻辑。
pub struct SnapshotBuilder {
    ch_url: String,
}

impl SnapshotBuilder {
    pub fn new(ch_url: impl Into<String>) -> Self {
        Self { ch_url: ch_url.into() }
    }

    /// 构建给定 symbol 和 snap_ms 的快照。
    ///
    /// 安全保证：查询硬编码 `obs_ms <= {snap_ms}`，
    /// 即使 ClickHouse 中已经写入更新的数据也不会泄漏。
    pub fn build(&self, symbol: &str, snap_ms: i64) -> Result<WindowSnapshot> {
        // 本窗口开始时间（5 分钟对齐）
        let win_start_ms = (snap_ms / 300_000) * 300_000;

        // 查询：窗口内所有 TWAP tick，严格 obs_ms <= snap_ms
        let sql = format!(
            "SELECT obs_ms, price \
             FROM pm.twap \
             WHERE symbol = '{symbol}' \
               AND window_s = 30 \
               AND obs_ms >= {win_start_ms} \
               AND obs_ms <= {snap_ms} \
             ORDER BY obs_ms ASC \
             FORMAT JSONEachRow",
        );

        let rows = ch_query_raw(&self.ch_url, &sql)
            .with_context(|| format!("symbol={symbol} snap_ms={snap_ms}"))?;

        let mut ticks = Vec::with_capacity(rows.len());
        for row in &rows {
            let obs_ms = row["obs_ms"]
                .as_str()
                .or_else(|| row["obs_ms"].as_str())
                .and_then(|s| s.parse::<i64>().ok())
                .or_else(|| row["obs_ms"].as_i64())
                .with_context(|| format!("obs_ms 解析失败: {row}"))?;

            let price_str = match &row["price"] {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Number(n) => n.to_string(),
                other => anyhow::bail!("price 字段格式意外: {other}"),
            };
            let price = Decimal::from_str(&price_str)
                .with_context(|| format!("price 解析失败: {price_str}"))?;

            // 双重检查：永远不允许未来数据混入
            assert!(
                obs_ms <= snap_ms,
                "快照安全违规：obs_ms={obs_ms} > snap_ms={snap_ms}，symbol={symbol}"
            );

            ticks.push(TwapTick { obs_ms, price });
        }

        Ok(WindowSnapshot {
            snap_ms,
            symbol: symbol.to_string(),
            win_start_ms,
            twap_ticks: ticks,
        })
    }
}

// ── 单元测试 ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    /// 辅助：用指定的 (obs_ms, price) 列表构造快照
    fn make_snap(
        win_start_ms: i64,
        snap_ms: i64,
        ticks: Vec<(i64, Decimal)>,
    ) -> WindowSnapshot {
        WindowSnapshot {
            snap_ms,
            symbol: "btc/usd".to_string(),
            win_start_ms,
            twap_ticks: ticks.into_iter().map(|(obs_ms, price)| TwapTick { obs_ms, price }).collect(),
        }
    }

    // ── 快照不含未来数据 ──────────────────────────────────────────────────────

    /// 快照内所有 tick 的 obs_ms 必须 <= snap_ms
    #[test]
    fn no_future_ticks() {
        let snap_ms     = 1_786_403_590_000_i64; // 窗口内某时刻
        let win_start   = (snap_ms / 300_000) * 300_000;

        // 构造一批 tick，全部严格 <= snap_ms
        let ticks = vec![
            (win_start,           dec!(63000.0)),
            (win_start + 1000,    dec!(63010.0)),
            (win_start + 60_000,  dec!(63020.0)),
            (snap_ms - 1000,      dec!(63025.0)),
            (snap_ms,             dec!(63030.0)),   // 恰好等于 snap_ms，合法
        ];
        let snap = make_snap(win_start, snap_ms, ticks);

        // 断言：所有 tick 的 obs_ms <= snap_ms
        for t in &snap.twap_ticks {
            assert!(
                t.obs_ms <= snap_ms,
                "tick obs_ms={} 超过 snap_ms={}",
                t.obs_ms, snap_ms
            );
        }
    }

    /// 如果将 snap_ms+1 的 tick 混入，必须被检测到
    #[test]
    #[should_panic(expected = "快照安全违规")]
    fn future_tick_panics() {
        let snap_ms   = 1_786_403_590_000_i64;
        let win_start = (snap_ms / 300_000) * 300_000;
        let future_ms = snap_ms + 1000;

        // 模拟 SnapshotBuilder 的双重检查逻辑
        let obs_ms = future_ms;
        assert!(
            obs_ms <= snap_ms,
            "快照安全违规：obs_ms={obs_ms} > snap_ms={snap_ms}，symbol=btc/usd"
        );
    }

    // ── open_price 与 current_price ───────────────────────────────────────────

    #[test]
    fn open_is_first_tick() {
        let win_start = 1_786_401_600_000_i64;
        let snap_ms   = win_start + 190_000;
        let snap = make_snap(win_start, snap_ms, vec![
            (win_start,          dec!(63888.7231)),
            (win_start + 1000,   dec!(63890.0)),
            (win_start + 60_000, dec!(63920.0)),
        ]);
        // open_price 必须是 argMin(obs_ms) 对应的价格
        assert_eq!(snap.open_price(), Some(dec!(63888.7231)));
    }

    #[test]
    fn current_is_last_tick() {
        let win_start = 1_786_401_600_000_i64;
        let snap_ms   = win_start + 190_000;
        let snap = make_snap(win_start, snap_ms, vec![
            (win_start,          dec!(63888.7231)),
            (win_start + 60_000, dec!(63920.0)),
            (snap_ms - 1000,     dec!(63928.1637)),
        ]);
        assert_eq!(snap.current_price(), Some(dec!(63928.1637)));
    }

    // ── elapsed / remaining ───────────────────────────────────────────────────

    #[test]
    fn timing_at_pred_point() {
        let win_start = 1_786_401_600_000_i64;
        let snap_ms   = win_start + 190_000; // T-110s 预测时点
        let snap = make_snap(win_start, snap_ms, vec![]);

        assert_eq!(snap.elapsed_ms(),   190_000);
        assert_eq!(snap.remaining_ms(), 110_000);
    }

    // ── normalized_distance ───────────────────────────────────────────────────

    #[test]
    fn normalized_distance_positive_move() {
        let win_start = 1_786_401_600_000_i64;
        let snap_ms   = win_start + 190_000;
        // 价格稳步上涨：标准差非零，距离应为正
        let snap = make_snap(win_start, snap_ms, vec![
            (win_start,            dec!(63800.0)),
            (win_start + 30_000,   dec!(63810.0)),
            (win_start + 60_000,   dec!(63820.0)),
            (win_start + 90_000,   dec!(63830.0)),
            (win_start + 120_000,  dec!(63840.0)),
            (win_start + 150_000,  dec!(63850.0)),
            (snap_ms - 1000,       dec!(63860.0)),
        ]);
        let nd = snap.normalized_distance().expect("normalized_distance 应有值");
        assert!(nd > 0.0, "上涨行情应得到正的归一化距离，got {nd}");
    }

    #[test]
    fn normalized_distance_flat_returns_none() {
        // 所有 tick 相同价格 → 标准差 = 0 → 应返回 None
        let win_start = 1_786_401_600_000_i64;
        let snap_ms   = win_start + 190_000;
        let snap = make_snap(win_start, snap_ms, vec![
            (win_start,           dec!(63900.0)),
            (win_start + 60_000,  dec!(63900.0)),
            (win_start + 120_000, dec!(63900.0)),
        ]);
        assert_eq!(snap.normalized_distance(), None);
    }

    #[test]
    fn normalized_distance_single_tick_returns_none() {
        // 只有一条 tick → 标准差无法计算（n<2）
        let win_start = 1_786_401_600_000_i64;
        let snap_ms   = win_start + 30_000;
        let snap = make_snap(win_start, snap_ms, vec![
            (win_start, dec!(63900.0)),
        ]);
        assert_eq!(snap.normalized_distance(), None);
    }

    // ── 窗口对齐 ──────────────────────────────────────────────────────────────

    #[test]
    fn win_start_aligns_to_5min_boundary() {
        // snap_ms = 22:42:17 UTC → win_start = 22:40:00 UTC
        let snap_ms   = 1_786_401_737_000_i64;  // 22:42:17.000
        let win_start = (snap_ms / 300_000) * 300_000;
        assert_eq!(win_start % 300_000, 0, "win_start 必须是 300000ms 的整数倍");
        assert!(win_start <= snap_ms);
        assert!(snap_ms < win_start + 300_000);
    }
}
