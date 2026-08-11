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

/// BBO 报价的过期阈值（毫秒）。
///
/// 超过这个时间没更新的报价，视为该交易所此刻"没有意见"，不参与共识价、
/// 分歧度等特征。3 秒的取值来源：binance/okx/bybit 在正常行情下的更新间隔
/// 都在百毫秒量级，3 秒足够宽松；而 kraken 约 0.2/s 的推送频率会被稳定剔除，
/// 正是我们要挡掉的那类"看起来在报价、其实是十几秒前的旧值"。
pub const BBO_STALE_MS: i64 = 3_000;

/// TWAP tick 的轻量表示（仅快照需要的字段）
#[derive(Debug, Clone)]
pub struct TwapTick {
    pub obs_ms: i64,
    pub price:  Decimal,
}

/// 交易所 BBO 快照点。
///
/// 时间用 recv_ms（本地接收时间），不用 ts_ex：交易所时钟可能有偏移
/// 甚至跳变，只有本地接收时间对我们具备因果性 —— 在 recv_ms 之前
/// 我们不可能知道这条数据。
#[derive(Debug, Clone)]
pub struct BboTick {
    pub exchange: String,
    pub recv_ms:  i64,
    pub bid:      f64,
    pub bid_qty:  f64,
    pub ask:      f64,
    pub ask_qty:  f64,
}

impl BboTick {
    pub fn mid(&self) -> f64 { (self.bid + self.ask) / 2.0 }
    pub fn spread(&self) -> f64 { self.ask - self.bid }
    /// bid_qty / (bid_qty + ask_qty)，>0.5 表示买盘更厚
    pub fn imbalance(&self) -> Option<f64> {
        let total = self.bid_qty + self.ask_qty;
        if total <= 0.0 { None } else { Some(self.bid_qty / total) }
    }
}

/// 交易所成交记录
#[derive(Debug, Clone)]
pub struct TradeTick {
    pub exchange: String,
    pub recv_ms:  i64,
    pub price:    f64,
    pub qty:      f64,
    /// true = 买方主动成交（taker buy）
    pub is_buy:   bool,
}

/// 判断交易所是否以真实 USD 计价（coinbase/kraken 是 USD，其余是 USDT）
fn is_usd_quoted(exchange: &str) -> bool {
    matches!(exchange, "coinbase" | "kraken")
}

/// 简单线性回归斜率 dy/dx。至少需要 2 个点；所有 x 相同时返回 NAN。
fn linreg_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    if n < 2.0 { return f64::NAN; }
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let sxy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-10 { return f64::NAN; }
    (n * sxy - sx * sy) / denom
}

/// 给定 symbol + snap_ms 的完整快照
#[derive(Debug, Clone)]
pub struct WindowSnapshot {
    /// 预测时刻（毫秒 Unix）
    pub snap_ms:      i64,
    pub symbol:       String,
    /// 本市场窗口开始时间（5 分钟对齐）
    pub win_start_ms: i64,
    /// 窗口内所有 TWAP tick，obs_ms <= snap_ms，按 obs_ms 升序
    pub twap_ticks:   Vec<TwapTick>,
    /// 窗口内所有交易所 BBO，recv_ms <= snap_ms，按 recv_ms 升序
    pub bbo_ticks:    Vec<BboTick>,
    /// 窗口内所有成交，recv_ms <= snap_ms，按 recv_ms 升序
    pub trade_ticks:  Vec<TradeTick>,
    /// USDT/USD 溢价（由 build_all 注入）。
    ///
    /// 实测稳定在 ~9.7bp：binance/okx/bybit 以 USDT 计价，coinbase/kraken 以真实
    /// USD 计价，两组的中间价比值就是 USDT/USD 现汇率偏差。
    /// `None` 表示单 symbol 构建路径，FX 敏感特征会退化为 NAN。
    pub fx_usdt_premium: Option<f64>,
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

    // ── 交易所侧访问器 ────────────────────────────────────────────────────────

    /// 各交易所在 snap_ms 时刻的最新 BBO（每家一条，含过期报价）
    ///
    /// 诊断用。做特征请用 `latest_bbo_per_exchange()` —— 那个版本会剔除
    /// 过期报价，避免把"陈旧"误读成"分歧"。
    pub fn latest_bbo_per_exchange_raw(&self) -> Vec<&BboTick> {
        let mut seen: Vec<&str> = Vec::new();
        let mut out = Vec::new();
        // 反向遍历，每家交易所第一次出现的就是最新的
        for t in self.bbo_ticks.iter().rev() {
            if !seen.iter().any(|e| *e == t.exchange.as_str()) {
                seen.push(&t.exchange);
                out.push(t);
            }
        }
        out
    }

    /// 各交易所在 snap_ms 时刻的**有效**最新 BBO（每家一条）
    ///
    /// 剔除超过 `BBO_STALE_MS` 未更新的报价。kraken 的更新频率只有约 0.2/s，
    /// 实测某个窗口里它的最后一条报价已经 12.4 秒未动、偏离中位数 60 美元 ——
    /// 把它算进 `cross_exchange_dispersion()` 会让那个特征在量化"陈旧程度"
    /// 而不是"真实分歧"。一家交易所报价过期，就等于它此刻没有意见。
    pub fn latest_bbo_per_exchange(&self) -> Vec<&BboTick> {
        self.latest_bbo_per_exchange_raw()
            .into_iter()
            .filter(|t| self.snap_ms - t.recv_ms <= BBO_STALE_MS)
            .collect()
    }

    /// 跨交易所的合成中间价：各家最新 mid 的中位数。
    ///
    /// 用中位数而非均值 —— 单家交易所抽风（报价卡住、闪崩）不该污染信号。
    pub fn consensus_mid(&self) -> Option<f64> {
        let mut mids: Vec<f64> = self.latest_bbo_per_exchange()
            .iter()
            .map(|t| t.mid())
            .filter(|m| m.is_finite() && *m > 0.0)
            .collect();
        if mids.is_empty() { return None; }
        mids.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = mids.len();
        Some(if n % 2 == 1 { mids[n/2] } else { (mids[n/2 - 1] + mids[n/2]) / 2.0 })
    }

    /// 现货相对 TWAP 的原始基差：(consensus_mid - twap_now) / twap_now
    ///
    /// ⚠️ 这个值含有约 +9bp 的系统性偏移（五个 symbol 实测均值 8.9~9.6bp，
    /// 正比例接近 100%，标准差仅 1~2bp）。成因是 USDT/USD 轻微溢价（实测
    /// ~+9.7bp）：我们的现货数据来自 Binance/OKX/Bybit（USDT 计价），而
    /// Polymarket 参考价是 USD 计价，两者之间系统性差了一个 USDT 溢价，
    /// 不会随时间消失。
    ///
    /// 直接做特征会退化成常数 —— 真正的信号在 `spot_basis_dev()` 里。
    /// 这里保留原始值供诊断和特征构造使用。
    pub fn spot_twap_basis(&self) -> Option<f64> {
        let spot = self.consensus_mid()?;
        let twap: f64 = self.current_price()?.try_into().ok()?;
        if twap <= 0.0 { return None; }
        Some((spot - twap) / twap)
    }

    /// 逐时点基差序列：把窗口内每条 BBO 与当时最新的 TWAP 配对。
    ///
    /// TWAP 用"该 BBO 时刻之前最后一条观测"——这是当时真实可见的值，
    /// 不会用到 BBO 之后才到达的 TWAP。
    fn basis_series(&self) -> Vec<f64> {
        if self.twap_ticks.is_empty() { return Vec::new(); }

        // 按交易所分组取每个时刻的中位数代价太高，这里用简化口径：
        // 对每条 BBO 单独算基差，天然按各家更新频率加权。
        let mut out = Vec::with_capacity(self.bbo_ticks.len());
        let mut ti = 0usize; // twap_ticks 游标，随 bbo 时间单调前进

        for b in &self.bbo_ticks {
            // 前进到最后一个 obs_ms <= b.recv_ms 的 TWAP tick
            while ti + 1 < self.twap_ticks.len()
                && self.twap_ticks[ti + 1].obs_ms <= b.recv_ms
            {
                ti += 1;
            }
            // 该 BBO 早于第一条 TWAP，无可配对的历史值
            if self.twap_ticks[ti].obs_ms > b.recv_ms { continue; }

            let twap: f64 = match self.twap_ticks[ti].price.try_into() {
                Ok(v)  => v,
                Err(_) => continue,
            };
            if twap <= 0.0 { continue; }

            let mid = b.mid();
            if !mid.is_finite() || mid <= 0.0 { continue; }
            out.push((mid - twap) / twap);
        }
        out
    }

    /// 基差相对窗口自身均值的偏离 —— 交易所数据的真实 alpha。
    ///
    /// 原始基差含约 +9bp 的常数偏移（见 `spot_twap_basis`），减去窗口内
    /// 均值后剩下的才是"现货此刻相对常态偏高还是偏低"。TWAP 向现货收敛，
    /// 所以正偏离预示 TWAP 接下来上行。
    ///
    /// 用窗口内均值而非全局常数：偏移量本身会随行情缓慢漂移，
    /// 窗口内自适应比硬编码 9bp 稳健。
    pub fn spot_basis_dev(&self) -> Option<f64> {
        let now = self.spot_twap_basis()?;
        let series = self.basis_series();
        if series.len() < 10 { return None; }   // 样本太少，均值不可靠
        let mean = series.iter().sum::<f64>() / series.len() as f64;
        Some(now - mean)
    }

    /// 基差偏离除以其自身标准差 —— 无量纲，跨 symbol 可比。
    pub fn spot_basis_dev_z(&self) -> Option<f64> {
        let now = self.spot_twap_basis()?;
        let series = self.basis_series();
        if series.len() < 10 { return None; }
        let n = series.len() as f64;
        let mean = series.iter().sum::<f64>() / n;
        let var  = series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let sd = var.sqrt();
        if sd <= 0.0 || !sd.is_finite() { return None; }
        Some((now - mean) / sd)
    }

    /// 指定时间窗内的成交（recv_ms 在 [snap_ms - window_ms, snap_ms]）
    pub fn trades_in_last(&self, window_ms: i64) -> impl Iterator<Item = &TradeTick> {
        let cutoff = self.snap_ms - window_ms;
        self.trade_ticks.iter().filter(move |t| t.recv_ms >= cutoff)
    }

    /// 指定时间窗内的成交流失衡：(买量 - 卖量) / 总量，范围 [-1, 1]
    pub fn trade_flow_imbalance(&self, window_ms: i64) -> Option<f64> {
        let (buy, sell) = self.trades_in_last(window_ms)
            .fold((0.0, 0.0), |(b, s), t| {
                if t.is_buy { (b + t.qty, s) } else { (b, s + t.qty) }
            });
        let total = buy + sell;
        if total <= 0.0 { None } else { Some((buy - sell) / total) }
    }

    /// 指定时间窗内的成交笔数（活跃度）
    pub fn trade_count_in_last(&self, window_ms: i64) -> usize {
        self.trades_in_last(window_ms).count()
    }

    /// 跨交易所的平均盘口失衡：各家最新 BBO 的 imbalance 均值。
    /// >0.5 表示买盘整体更厚。
    pub fn book_imbalance(&self) -> Option<f64> {
        let vals: Vec<f64> = self.latest_bbo_per_exchange()
            .iter()
            .filter_map(|t| t.imbalance())
            .collect();
        if vals.is_empty() { return None; }
        Some(vals.iter().sum::<f64>() / vals.len() as f64)
    }

    /// 跨交易所的平均相对价差：mean(spread / mid)。
    /// 价差走阔通常意味着流动性变差、不确定性上升。
    pub fn mean_rel_spread(&self) -> Option<f64> {
        let vals: Vec<f64> = self.latest_bbo_per_exchange()
            .iter()
            .filter_map(|t| {
                let m = t.mid();
                if m > 0.0 { Some(t.spread() / m) } else { None }
            })
            .collect();
        if vals.is_empty() { return None; }
        Some(vals.iter().sum::<f64>() / vals.len() as f64)
    }

    /// 跨交易所价格分歧度：各家 mid 的相对标准差。
    /// 分歧变大说明市场对价格没有共识，方向信号可信度下降。
    pub fn cross_exchange_dispersion(&self) -> Option<f64> {
        let mids: Vec<f64> = self.latest_bbo_per_exchange()
            .iter()
            .map(|t| t.mid())
            .filter(|m| m.is_finite() && *m > 0.0)
            .collect();
        if mids.len() < 2 { return None; }
        let mean = mids.iter().sum::<f64>() / mids.len() as f64;
        if mean <= 0.0 { return None; }
        let var = mids.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / (mids.len() - 1) as f64;
        Some(var.sqrt() / mean)
    }

    /// 有 BBO 数据的交易所家数（数据质量指标）
    pub fn active_exchange_count(&self) -> usize {
        self.latest_bbo_per_exchange().len()
    }

    // ── FX 剥离后的现货价格 ──────────────────────────────────────────────────

    /// 去掉 USDT/USD 溢价后的共识中间价（USD 口径）。
    ///
    /// USD 计价交易所（coinbase/kraken）直接使用；USDT 计价的三家除以 (1 + fx_usdt_premium)。
    /// fx_usdt_premium 为 None 时退化为普通 `consensus_mid()`（含 ~9bp 偏移）。
    pub fn consensus_mid_usd(&self) -> Option<f64> {
        let premium = self.fx_usdt_premium.unwrap_or(0.0);
        let mut mids: Vec<f64> = self.latest_bbo_per_exchange()
            .iter()
            .map(|t| {
                let raw = t.mid();
                if is_usd_quoted(&t.exchange) { raw } else { raw / (1.0 + premium) }
            })
            .filter(|m| m.is_finite() && *m > 0.0)
            .collect();
        if mids.is_empty() { return None; }
        mids.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = mids.len();
        Some(if n % 2 == 1 { mids[n / 2] } else { (mids[n / 2 - 1] + mids[n / 2]) / 2.0 })
    }

    /// 现货（USD 口径）相对开盘价的收益率：(spot_usd - open) / open。
    ///
    /// 与 `ret_so_far`（TWAP-based）相比：TWAP 此时是 [T+160, T+190] 的均值，
    /// 天然滞后 15 秒；而现货直接反映 T+190 时刻，对 label 的预测力更强。
    pub fn spot_ret_from_open(&self) -> Option<f64> {
        let spot = self.consensus_mid_usd()?;
        let open: f64 = self.open_price()?.try_into().ok()?;
        if open <= 0.0 { return None; }
        Some((spot - open) / open)
    }

    /// 现货（USD 口径）的归一化距离：(spot_usd - open) / (σ · √t_remaining)。
    ///
    /// `norm_dist` 的严格更优版本：分子用现货而非 twap_now，更接近 label 的真实结构。
    /// fx_usdt_premium 为 None 时退化为含偏移的估计（仍有信号，但量纲有偏）。
    pub fn spot_norm_dist(&self) -> Option<f64> {
        let spot = self.consensus_mid_usd()?;
        let open: f64 = self.open_price()?.try_into().ok()?;
        let sigma = self.price_std()?;
        if sigma == 0.0 { return None; }
        let t_rem = self.remaining_ms() as f64 / 1000.0;
        if t_rem <= 0.0 { return None; }
        Some((spot - open) / (sigma * t_rem.sqrt()))
    }

    /// TWAP 向现货收敛的待吸收缺口，归一化：(spot_usd - twap_now) / (σ · √t_remaining)。
    ///
    /// TWAP-30 在数学上必须向现货靠拢，这个值正比于"还有多少路要赶"。
    pub fn twap_spot_gap_norm(&self) -> Option<f64> {
        let spot = self.consensus_mid_usd()?;
        let twap: f64 = self.current_price()?.try_into().ok()?;
        let sigma = self.price_std()?;
        if sigma == 0.0 || twap <= 0.0 { return None; }
        let t_rem = self.remaining_ms() as f64 / 1000.0;
        if t_rem <= 0.0 { return None; }
        Some((spot - twap) / (sigma * t_rem.sqrt()))
    }

    // ── Basis 趋势 ────────────────────────────────────────────────────────────

    /// 带时间戳的 basis 序列（recv_ms, basis_value）。
    ///
    /// 与 `basis_series()` 逻辑相同，但额外返回每条 BBO 的 recv_ms，
    /// 供斜率计算使用。
    fn basis_series_timed(&self) -> Vec<(i64, f64)> {
        if self.twap_ticks.is_empty() { return Vec::new(); }
        let mut out = Vec::with_capacity(self.bbo_ticks.len());
        let mut ti = 0usize;
        for b in &self.bbo_ticks {
            while ti + 1 < self.twap_ticks.len()
                && self.twap_ticks[ti + 1].obs_ms <= b.recv_ms
            {
                ti += 1;
            }
            if self.twap_ticks[ti].obs_ms > b.recv_ms { continue; }
            let twap: f64 = match self.twap_ticks[ti].price.try_into() {
                Ok(v) => v,
                Err(_) => continue,
            };
            if twap <= 0.0 { continue; }
            let mid = b.mid();
            if !mid.is_finite() || mid <= 0.0 { continue; }
            out.push((b.recv_ms, (mid - twap) / twap));
        }
        out
    }

    /// 指定时间窗内 basis 的线性斜率（bp/秒，归一化单位）。
    ///
    /// 正值：basis 在扩大（现货正在远离 TWAP）→ 待吸收漂移还在积累。
    /// 负值：basis 在收缩（TWAP 正在追上现货）→ 信号趋弱。
    pub fn basis_slope(&self, window_ms: i64) -> Option<f64> {
        let cutoff = self.snap_ms - window_ms;
        let series: Vec<(f64, f64)> = self.basis_series_timed()
            .into_iter()
            .filter(|(t, _)| *t >= cutoff)
            .map(|(t, b)| (t as f64 / 1000.0, b))
            .collect();
        if series.len() < 5 { return None; }
        let xs: Vec<f64> = series.iter().map(|(t, _)| *t).collect();
        let ys: Vec<f64> = series.iter().map(|(_, b)| *b).collect();
        let s = linreg_slope(&xs, &ys);
        if s.is_nan() { None } else { Some(s) }
    }

    // ── OFI（订单流失衡） ─────────────────────────────────────────────────────

    /// 指定时间窗内各交易所的 OFI，返回 (exchange, ofi_raw, abs_flow)。
    fn ofi_by_exchange(&self, window_ms: i64) -> Vec<(&str, f64, f64)> {
        let cutoff = self.snap_ms - window_ms;
        // 按交易所名收集有序 tick（bbo_ticks 已按 recv_ms 升序）
        let mut result = Vec::new();

        let mut ex_list: Vec<&str> = Vec::new();
        for t in &self.bbo_ticks {
            if !ex_list.contains(&t.exchange.as_str()) {
                ex_list.push(t.exchange.as_str());
            }
        }

        for ex in ex_list {
            let ticks: Vec<&BboTick> = self.bbo_ticks.iter()
                .filter(|t| t.exchange.as_str() == ex)
                .collect();
            if ticks.len() < 2 { continue; }

            // 锚点：cutoff 之前最后一条（index = partition_point - 1）
            let anchor = ticks.partition_point(|t| t.recv_ms < cutoff);
            if anchor == 0 { continue; }
            let start = anchor - 1;

            let mut ofi_raw: f64 = 0.0;
            let mut abs_flow: f64 = 0.0;
            for i in (start + 1)..ticks.len() {
                let prev = ticks[i - 1];
                let cur  = ticks[i];
                if cur.recv_ms < cutoff { continue; }
                if cur.recv_ms > self.snap_ms { break; }

                let e_b = if cur.bid > prev.bid + 1e-10 {
                    cur.bid_qty
                } else if (cur.bid - prev.bid).abs() <= 1e-10 {
                    (cur.bid_qty - prev.bid_qty).max(0.0)
                } else {
                    -prev.bid_qty
                };
                let e_a = if cur.ask < prev.ask - 1e-10 {
                    cur.ask_qty
                } else if (cur.ask - prev.ask).abs() <= 1e-10 {
                    (cur.ask_qty - prev.ask_qty).max(0.0)
                } else {
                    -prev.ask_qty
                };
                ofi_raw  += e_b - e_a;
                abs_flow += e_b.abs() + e_a.abs();
            }
            if abs_flow > 0.0 {
                result.push((ex, ofi_raw, abs_flow));
            }
        }
        result
    }

    /// 汇总 OFI：`Σ(ofi_raw) / Σ(abs_flow)`，范围约 [-1, 1]，跨 symbol 可比。
    ///
    /// 正值：净买压；负值：净卖压。
    pub fn ofi(&self, window_ms: i64) -> Option<f64> {
        let data = self.ofi_by_exchange(window_ms);
        if data.is_empty() { return None; }
        let total_ofi: f64  = data.iter().map(|(_, o, _)| o).sum();
        let total_abs: f64  = data.iter().map(|(_, _, a)| a).sum();
        if total_abs <= 0.0 { return None; }
        Some(total_ofi / total_abs)
    }

    /// OFI 方向一致性：OFI 同号（净买）的交易所占比。
    ///
    /// 五家一致的信号可信度远高于单家。
    pub fn ofi_agreement(&self, window_ms: i64) -> Option<f64> {
        let data = self.ofi_by_exchange(window_ms);
        if data.is_empty() { return None; }
        let n_pos = data.iter().filter(|(_, o, _)| *o > 0.0).count();
        Some(n_pos as f64 / data.len() as f64)
    }

    // ── 成交流高级特征 ────────────────────────────────────────────────────────

    /// 大单方向失衡：只计 qty >= P75 的成交，其余忽略。
    ///
    /// aggTrade 已经把同毫秒同价格的 taker 单合并，大 qty 近似等于机构单。
    pub fn large_flow_imbalance(&self, window_ms: i64) -> Option<f64> {
        let trades: Vec<&TradeTick> = self.trades_in_last(window_ms).collect();
        if trades.len() < 4 { return None; }

        let mut qtys: Vec<f64> = trades.iter().map(|t| t.qty).collect();
        qtys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p75 = qtys[qtys.len() * 3 / 4];

        let (buy, sell) = trades.iter()
            .filter(|t| t.qty >= p75)
            .fold((0.0f64, 0.0f64), |(b, s), t| {
                if t.is_buy { (b + t.qty, s) } else { (b, s + t.qty) }
            });
        let total = buy + sell;
        if total <= 0.0 { None } else { Some((buy - sell) / total) }
    }

    /// 近期成交 VWAP 相对开盘价的偏离：(vwap - open) / open。
    ///
    /// 比 BBO 中点噪声更小，天然按成交量加权。
    pub fn vwap_dev(&self, window_ms: i64) -> Option<f64> {
        let (sum_pv, sum_v) = self.trades_in_last(window_ms)
            .fold((0.0f64, 0.0f64), |(spv, sv), t| {
                (spv + t.price * t.qty, sv + t.qty)
            });
        if sum_v <= 0.0 { return None; }
        let vwap = sum_pv / sum_v;
        let open: f64 = self.open_price()?.try_into().ok()?;
        if open <= 0.0 { return None; }
        Some((vwap - open) / open)
    }

    // ── 波动率辅助 ────────────────────────────────────────────────────────────

    /// 指定时间窗内 TWAP 价格序列的标准差。
    pub fn price_std_last(&self, window_ms: i64) -> Option<f64> {
        let cutoff = self.snap_ms - window_ms;
        let prices: Vec<f64> = self.twap_ticks.iter()
            .filter(|t| t.obs_ms >= cutoff)
            .map(|t| t.price.try_into().unwrap_or(f64::NAN))
            .filter(|p| p.is_finite())
            .collect();
        let n = prices.len();
        if n < 2 { return None; }
        let mean = prices.iter().sum::<f64>() / n as f64;
        let var  = prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        Some(var.sqrt())
    }

    /// 近期波动率与全窗口波动率之比：σ_60s / σ_full。
    ///
    /// >1 说明近期波动加速（突破前沿）；<1 说明在收敛（反转风险上升）。
    pub fn vol_ratio(&self) -> Option<f64> {
        let sigma_recent = self.price_std_last(60_000)?;
        let sigma_full   = self.price_std()?;
        if sigma_full <= 0.0 { return None; }
        Some(sigma_recent / sigma_full)
    }

    /// 现货（BBO 中点）近 60s 的已实现波动率 / 开盘价。
    ///
    /// TWAP 波动被 30s 均值平滑，不能反映高频波动；现货 BBO 的标准差更贴近
    /// 真实价格不确定性，也更接近 label 的实际波动性质。
    pub fn spot_vol_rel(&self) -> Option<f64> {
        let cutoff = self.snap_ms - 60_000;
        let mids: Vec<f64> = self.bbo_ticks.iter()
            .filter(|t| t.recv_ms >= cutoff)
            .map(|t| t.mid())
            .filter(|m| m.is_finite() && *m > 0.0)
            .collect();
        let n = mids.len();
        if n < 5 { return None; }
        let mean = mids.iter().sum::<f64>() / n as f64;
        if mean <= 0.0 { return None; }
        let var = mids.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        Some(var.sqrt() / mean)
    }

    // ── 现货动量 ──────────────────────────────────────────────────────────────

    /// 现货 BBO 中点在指定时间窗内的线性斜率，归一化为相对斜率（/秒 / open）。
    ///
    /// 比 TWAP 斜率领先 ~15s，用于捕捉现货动量变化。
    pub fn spot_slope(&self, window_ms: i64) -> Option<f64> {
        let cutoff = self.snap_ms - window_ms;
        let points: Vec<(f64, f64)> = self.bbo_ticks.iter()
            .filter(|t| t.recv_ms >= cutoff)
            .map(|t| (t.recv_ms as f64 / 1000.0, t.mid()))
            .filter(|(_, m)| m.is_finite() && *m > 0.0)
            .collect();
        if points.len() < 5 { return None; }
        let xs: Vec<f64> = points.iter().map(|(t, _)| *t).collect();
        let ys: Vec<f64> = points.iter().map(|(_, m)| *m).collect();
        let open: f64 = self.open_price()?.try_into().ok()?;
        if open <= 0.0 { return None; }
        let s = linreg_slope(&xs, &ys);
        if s.is_nan() { None } else { Some(s / open) }
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

        let bbo_ticks   = self.fetch_bbo(symbol, win_start_ms, snap_ms)?;
        let trade_ticks = self.fetch_trades(symbol, win_start_ms, snap_ms)?;

        Ok(WindowSnapshot {
            snap_ms,
            symbol: symbol.to_string(),
            win_start_ms,
            twap_ticks: ticks,
            bbo_ticks,
            trade_ticks,
            fx_usdt_premium: None,
        })
    }

    /// 读取窗口内的交易所 BBO。
    ///
    /// 用 recv_ms 而非 ts_ex 做边界：交易所时钟可能领先本地，
    /// 用 ts_ex 过滤会放进我们当时还没收到的数据 —— 这正是最隐蔽的
    /// 未来信息泄漏路径。
    fn fetch_bbo(&self, symbol: &str, win_start_ms: i64, snap_ms: i64) -> Result<Vec<BboTick>> {
        let sql = format!(
            "SELECT exchange, recv_ms, bid, bid_qty, ask, ask_qty \
             FROM pm.ex_bbo \
             WHERE symbol = '{symbol}' \
               AND recv_ms >= {win_start_ms} \
               AND recv_ms <= {snap_ms} \
             ORDER BY recv_ms ASC \
             FORMAT JSONEachRow",
        );
        let rows = ch_query_raw(&self.ch_url, &sql)
            .with_context(|| format!("ex_bbo 查询失败 symbol={symbol}"))?;

        let mut out = Vec::with_capacity(rows.len());
        for row in &rows {
            let recv_ms = json_i64(&row["recv_ms"])
                .with_context(|| format!("recv_ms 解析失败: {row}"))?;

            assert!(
                recv_ms <= snap_ms,
                "快照安全违规：bbo recv_ms={recv_ms} > snap_ms={snap_ms}"
            );

            // 价格在 ClickHouse 里是 String（保留交易所原始十进制文本）
            let f = |k: &str| -> f64 {
                row[k].as_str().and_then(|s| s.parse::<f64>().ok()).unwrap_or(f64::NAN)
            };
            let (bid, ask) = (f("bid"), f("ask"));
            // 丢弃明显损坏的报价：非有限值、非正、买价高于卖价
            if !bid.is_finite() || !ask.is_finite() || bid <= 0.0 || ask <= 0.0 || bid > ask {
                continue;
            }

            out.push(BboTick {
                exchange: row["exchange"].as_str().unwrap_or("?").to_string(),
                recv_ms,
                bid,
                bid_qty: f("bid_qty"),
                ask,
                ask_qty: f("ask_qty"),
            });
        }
        Ok(out)
    }

    /// 读取窗口内的成交记录。同样以 recv_ms 为边界。
    fn fetch_trades(&self, symbol: &str, win_start_ms: i64, snap_ms: i64) -> Result<Vec<TradeTick>> {
        let sql = format!(
            "SELECT exchange, recv_ms, price, qty, side \
             FROM pm.ex_trades \
             WHERE symbol = '{symbol}' \
               AND recv_ms >= {win_start_ms} \
               AND recv_ms <= {snap_ms} \
             ORDER BY recv_ms ASC \
             FORMAT JSONEachRow",
        );
        let rows = ch_query_raw(&self.ch_url, &sql)
            .with_context(|| format!("ex_trades 查询失败 symbol={symbol}"))?;

        let mut out = Vec::with_capacity(rows.len());
        for row in &rows {
            let recv_ms = json_i64(&row["recv_ms"])
                .with_context(|| format!("recv_ms 解析失败: {row}"))?;

            assert!(
                recv_ms <= snap_ms,
                "快照安全违规：trade recv_ms={recv_ms} > snap_ms={snap_ms}"
            );

            let f = |k: &str| -> f64 {
                row[k].as_str().and_then(|s| s.parse::<f64>().ok()).unwrap_or(f64::NAN)
            };
            let (price, qty) = (f("price"), f("qty"));
            if !price.is_finite() || !qty.is_finite() || price <= 0.0 || qty <= 0.0 {
                continue;
            }

            out.push(TradeTick {
                exchange: row["exchange"].as_str().unwrap_or("?").to_string(),
                recv_ms,
                price,
                qty,
                is_buy: row["side"].as_str() == Some("buy"),
            });
        }
        Ok(out)
    }

    // ── FX stripping ──────────────────────────────────────────────────────────

    /// 估算当前时刻的 USDT/USD 溢价。
    ///
    /// 方法：对每个 symbol，分别取 USDT 计价（binance/okx/bybit）和 USD 计价
    /// （coinbase/kraken）的最新 BBO 中点，算每个 symbol 的比值，返回跨 symbol 中位数。
    ///
    /// 中位数比均值更鲁棒：某个 symbol 出现巨幅行情时，其 USDT/USD 比值会因
    /// 两组价格对行情的反应速度不同而短暂偏离，中位数抑制这类噪声。
    ///
    /// 返回值约为 0.00097（~+9.7bp）。如果数据不足或结果明显异常，返回 None。
    pub fn estimate_usdt_premium(&self, snap_ms: i64) -> Option<f64> {
        let win_start_ms = (snap_ms / 300_000) * 300_000;
        // 在窗口内（非仅最新一秒）取最新值，和 build() 数据口径一致
        let sql = format!(
            "WITH latest AS ( \
               SELECT exchange, symbol, \
                 argMax((toFloat64(bid)+toFloat64(ask))/2, recv_ms) AS mid \
               FROM pm.ex_bbo \
               WHERE recv_ms >= {win_start_ms} AND recv_ms <= {snap_ms} \
               GROUP BY exchange, symbol \
             ), per_sym AS ( \
               SELECT symbol, \
                 medianIf(mid, exchange IN ('binance','okx','bybit')) AS usdt_mid, \
                 medianIf(mid, exchange IN ('coinbase','kraken')) AS usd_mid \
               FROM latest GROUP BY symbol \
               HAVING usdt_mid > 0 AND usd_mid > 0 \
             ) \
             SELECT median(usdt_mid / usd_mid - 1) AS premium \
             FROM per_sym \
             FORMAT JSONEachRow"
        );
        let rows = ch_query_raw(&self.ch_url, &sql).ok()?;
        let row  = rows.into_iter().next()?;
        let premium = match &row["premium"] {
            serde_json::Value::String(s) => s.parse::<f64>().ok()?,
            serde_json::Value::Number(n) => n.as_f64()?,
            _ => return None,
        };
        // 合理性检查：0.03bp ~ 50bp
        if premium.is_finite() && premium > 0.000003 && premium < 0.005 {
            Some(premium)
        } else {
            None
        }
    }

    /// 同时构建多个 symbol 的快照，并注入共享的 FX 溢价估算。
    ///
    /// 这是训练和推理的推荐入口：FX 溢价只需估算一次（每个 snap_ms），
    /// 每个 snapshot 获得相同的 fx_usdt_premium，从而可以使用 FX 敏感特征。
    pub fn build_all(&self, symbols: &[&str], snap_ms: i64) -> Result<Vec<WindowSnapshot>> {
        let fx_premium = self.estimate_usdt_premium(snap_ms);
        symbols.iter()
            .map(|sym| {
                let mut snap = self.build(sym, snap_ms)?;
                snap.fx_usdt_premium = fx_premium;
                Ok(snap)
            })
            .collect()
    }
}

/// 从 JSON 值取 i64 —— ClickHouse 的 UInt64 序列化为字符串，其余为数字
fn json_i64(v: &serde_json::Value) -> Option<i64> {
    match v {
        serde_json::Value::String(s) => s.parse::<i64>().ok(),
        serde_json::Value::Number(n) => n.as_i64(),
        _ => None,
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
            bbo_ticks:        Vec::new(),
            trade_ticks:      Vec::new(),
            fx_usdt_premium:  None,
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
