/// 特征超集：给定 WindowSnapshot，计算所有候选特征，输出 FeatureVector。
///
/// 设计原则：
///   1. 纯函数 —— 只依赖 snap，不读外部状态，训练/推理路径完全共用同一份代码。
///   2. 缺失安全 —— tick 不足时用 f64::NAN 占位，保持向量维度固定，
///      训练时由调用方决定是否过滤掉含 NAN 的行。
///   3. 显式命名 —— feature_names 与 features 严格同序，方便 SHAP 可视化。
///
/// 特征分组：
///   A. 位置类（价格在窗口内的相对位置）
///   B. 动量类（短期价格变化速率）
///   C. 结构类（基于布朗运动的归一化距离）
///   D. 质量类（数据稳健性指标，不直接预测方向，但可用于样本加权）

use crate::snapshot::WindowSnapshot;
use crate::types::FeatureVector;

// ── 内部辅助 ──────────────────────────────────────────────────────────────────

/// 取 `lookback_ms` 毫秒之前的价格：即 obs_ms <= (snap_ms - lookback_ms)
/// 的最后一条 tick。窗口内没有那么早的 tick 时返回 NAN。
fn price_n_ms_ago(snap: &WindowSnapshot, lookback_ms: i64) -> f64 {
    let cutoff = snap.snap_ms - lookback_ms;
    // ticks 按 obs_ms 升序；反向找第一个 obs_ms <= cutoff 的，
    // 即 cutoff 时刻或之前最新的那条。
    snap.twap_ticks.iter()
        .rev()
        .find(|t| t.obs_ms <= cutoff)
        .map(|t| t.price.try_into().unwrap_or(f64::NAN))
        .unwrap_or(f64::NAN)
}

/// 线性斜率（每秒价格变化量），用于动量特征。
/// 至少需要 2 个点；点不足或所有 x 相同时返回 NAN。
fn slope_per_sec(xs: &[f64], ys: &[f64]) -> f64 {
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

// ── 公共接口 ──────────────────────────────────────────────────────────────────

/// 从快照计算完整特征向量。
/// 向量维度固定为 `FEATURE_COUNT`，顺序与 `FEATURE_NAMES` 对应。
pub fn compute_features(snap: &WindowSnapshot) -> FeatureVector {
    let mut features: Vec<f64> = Vec::with_capacity(FEATURE_COUNT);

    let open: f64 = snap.open_price()
        .and_then(|p| p.try_into().ok())
        .unwrap_or(f64::NAN);
    let current: f64 = snap.current_price()
        .and_then(|p| p.try_into().ok())
        .unwrap_or(f64::NAN);
    let elapsed_s  = snap.elapsed_ms()   as f64 / 1000.0;
    let _remaining_s = snap.remaining_ms() as f64 / 1000.0;

    // ── A. 位置类 ─────────────────────────────────────────────────────────────

    // A1: 窗口内收益率 (current - open) / open，反映已走过的幅度
    let ret_so_far = if open.is_nan() || open == 0.0 {
        f64::NAN
    } else {
        (current - open) / open
    };
    features.push(ret_so_far);

    // A2: 价格在窗口内最高 / 最低之间的位置（0 = 最低，1 = 最高）
    let (lo, hi) = snap.twap_ticks.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), t| {
        let p: f64 = t.price.try_into().unwrap_or(f64::NAN);
        (lo.min(p), hi.max(p))
    });
    let price_position = if (hi - lo).abs() < 1e-10 || lo.is_infinite() {
        f64::NAN
    } else {
        (current - lo) / (hi - lo)
    };
    features.push(price_position);

    // A3: 距窗口最高价的归一化偏差
    let dist_from_hi = if hi.is_infinite() || open == 0.0 {
        f64::NAN
    } else {
        (hi - current) / open
    };
    features.push(dist_from_hi);

    // A4: 距窗口最低价的归一化偏差
    let dist_from_lo = if lo.is_infinite() || open == 0.0 {
        f64::NAN
    } else {
        (current - lo) / open
    };
    features.push(dist_from_lo);

    // ── B. 动量类 ─────────────────────────────────────────────────────────────

    // B1-B3: 最近 30s / 60s / 90s 的简单收益率
    for lookback_ms in [30_000_i64, 60_000, 90_000] {
        let past = price_n_ms_ago(snap, lookback_ms);
        let ret = if past.is_nan() || past == 0.0 || current.is_nan() {
            f64::NAN
        } else {
            (current - past) / past
        };
        features.push(ret);
    }

    // B4: 全窗口线性斜率（每秒收益率）
    let slope_all = {
        let xs: Vec<f64> = snap.twap_ticks.iter()
            .map(|t| (t.obs_ms - snap.win_start_ms) as f64 / 1000.0)
            .collect();
        let ys: Vec<f64> = snap.twap_ticks.iter()
            .map(|t| t.price.try_into().unwrap_or(f64::NAN))
            .collect();
        let s = slope_per_sec(&xs, &ys);
        // 归一化：除以 open，转为相对斜率
        if s.is_nan() || open == 0.0 { f64::NAN } else { s / open }
    };
    features.push(slope_all);

    // B5: 最近 60s 内的线性斜率（捕捉近期动量）
    let slope_recent = {
        let cutoff = snap.snap_ms - 60_000;
        let recent: Vec<&crate::snapshot::TwapTick> = snap.twap_ticks.iter()
            .filter(|t| t.obs_ms >= cutoff)
            .collect();
        let xs: Vec<f64> = recent.iter()
            .map(|t| (t.obs_ms - snap.win_start_ms) as f64 / 1000.0)
            .collect();
        let ys: Vec<f64> = recent.iter()
            .map(|t| t.price.try_into().unwrap_or(f64::NAN))
            .collect();
        let s = slope_per_sec(&xs, &ys);
        if s.is_nan() || open == 0.0 { f64::NAN } else { s / open }
    };
    features.push(slope_recent);

    // ── C. 结构类（布朗运动归一化）────────────────────────────────────────────

    // C1: normalized_distance = (P_now - P_open) / (σ · √t_remaining_s)
    features.push(snap.normalized_distance().unwrap_or(f64::NAN));

    // C2: normalized_distance 的平方（捕捉非线性，方向无关的偏移幅度）
    let nd = snap.normalized_distance().unwrap_or(f64::NAN);
    features.push(if nd.is_nan() { f64::NAN } else { nd * nd });

    // C3: 时间分数：已用时间 / 总窗口时长（0 ~ 1）
    let time_frac = elapsed_s / 300.0;
    features.push(time_frac);

    // C4: √时间分数（使时间维度与布朗运动的 √t 缩放对齐）
    features.push(time_frac.sqrt());

    // C5: 已实现波动率（annualized 不重要，相对值即可）：标准差 / open
    let vol = snap.price_std().unwrap_or(f64::NAN);
    let rel_vol = if vol.is_nan() || open == 0.0 { f64::NAN } else { vol / open };
    features.push(rel_vol);

    // ── D. 质量类（不直接预测方向，用于样本加权 / 过滤）────────────────────

    // D1: tick 密度（tick/秒），反映数据完整性
    let tick_density = if elapsed_s > 0.0 {
        snap.tick_count() as f64 / elapsed_s
    } else {
        f64::NAN
    };
    features.push(tick_density);

    // D2: 最近 tick 距 snap_ms 的延迟（秒），越小越新鲜
    let staleness_s = snap.twap_ticks.last().map(|t| {
        (snap.snap_ms - t.obs_ms).max(0) as f64 / 1000.0
    }).unwrap_or(f64::NAN);
    features.push(staleness_s);

    // ── E. 交易所类（现货领先 TWAP 的信息）──────────────────────────────────
    //
    // 核心逻辑：TWAP30 是 30 秒滞后平均，现货此刻的价格包含了 TWAP
    // 尚未吸收的信息。TWAP 在数学上必然向现货收敛，所以现货-TWAP 的
    // 价差预示 TWAP 接下来 30 秒的走向 —— 而收盘价正是由 TWAP 决定的。
    //
    // 但原始基差不能直接用：实测五个 symbol 的基差均值都稳定在 +9bp 附近
    // （标准差仅 1~2bp，正比例接近 100%），这是 Chainlink 价格构成与我们
    // 的买卖中点定义之间的口径差，不含方向信息。真正的信号是"此刻的基差
    // 相对本窗口常态偏高还是偏低"，所以下面两个特征都做了去均值处理。

    // E1: 基差相对窗口内均值的偏离
    let basis_dev = snap.spot_basis_dev().unwrap_or(f64::NAN);
    features.push(basis_dev);

    // E2: 同一偏离除以其自身标准差 —— 无量纲，跨 symbol / 跨波动环境可比
    features.push(snap.spot_basis_dev_z().unwrap_or(f64::NAN));

    // E3-E5: 30s / 60s / 全窗口的成交流失衡，(买量-卖量)/总量 ∈ [-1,1]
    for window_ms in [30_000_i64, 60_000] {
        features.push(snap.trade_flow_imbalance(window_ms).unwrap_or(f64::NAN));
    }
    features.push(snap.trade_flow_imbalance(snap.elapsed_ms().max(1)).unwrap_or(f64::NAN));

    // E6: 盘口失衡（各家最新 BBO 的 bid_qty 占比均值），>0.5 买盘更厚
    features.push(snap.book_imbalance().unwrap_or(f64::NAN));

    // E7: 平均相对价差 —— 走阔意味着流动性变差
    features.push(snap.mean_rel_spread().unwrap_or(f64::NAN));

    // E8: 跨交易所价格分歧度 —— 分歧大时方向信号可信度下降
    features.push(snap.cross_exchange_dispersion().unwrap_or(f64::NAN));

    // E9: 最近 30 秒成交笔数，取对数压缩量级差异（不同 symbol 差几个数量级）
    let tc30 = snap.trade_count_in_last(30_000) as f64;
    features.push((1.0 + tc30).ln());

    // E10: 有 BBO 数据的交易所家数（质量类，用于判断 E1-E8 的可信度）
    features.push(snap.active_exchange_count() as f64);

    debug_assert_eq!(
        features.len(), FEATURE_COUNT,
        "features 向量长度 {} 与 FEATURE_COUNT {} 不一致",
        features.len(), FEATURE_COUNT
    );

    FeatureVector {
        symbol:        snap.symbol.clone(),
        snap_ms:       snap.snap_ms,
        features,
        feature_names: FEATURE_NAMES.iter().map(|s| s.to_string()).collect(),
    }
}

// ── 单元测试 ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::{BboTick, TradeTick, TwapTick, WindowSnapshot};
    use rust_decimal_macros::dec;

    fn make_snap(win_start_ms: i64, snap_ms: i64, ticks: Vec<(i64, rust_decimal::Decimal)>) -> WindowSnapshot {
        WindowSnapshot {
            snap_ms,
            symbol: "btc/usd".to_string(),
            win_start_ms,
            twap_ticks: ticks.into_iter().map(|(obs_ms, price)| TwapTick { obs_ms, price }).collect(),
            bbo_ticks:   Vec::new(),
            trade_ticks: Vec::new(),
        }
    }

    /// 辅助：构造带交易所数据的快照
    fn make_snap_with_ex(
        win_start_ms: i64,
        snap_ms: i64,
        ticks: Vec<(i64, rust_decimal::Decimal)>,
        bbos: Vec<BboTick>,
        trades: Vec<TradeTick>,
    ) -> WindowSnapshot {
        WindowSnapshot {
            snap_ms,
            symbol: "btc/usd".to_string(),
            win_start_ms,
            twap_ticks: ticks.into_iter().map(|(obs_ms, price)| TwapTick { obs_ms, price }).collect(),
            bbo_ticks:   bbos,
            trade_ticks: trades,
        }
    }

    fn bbo(ex: &str, recv_ms: i64, bid: f64, bid_qty: f64, ask: f64, ask_qty: f64) -> BboTick {
        BboTick { exchange: ex.to_string(), recv_ms, bid, bid_qty, ask, ask_qty }
    }

    fn trade(ex: &str, recv_ms: i64, price: f64, qty: f64, is_buy: bool) -> TradeTick {
        TradeTick { exchange: ex.to_string(), recv_ms, price, qty, is_buy }
    }

    // ── 向量维度固定 ──────────────────────────────────────────────────────────

    #[test]
    fn feature_vector_length_matches_count() {
        let win_start = 1_786_401_600_000_i64;
        let snap_ms   = win_start + 190_000;
        let snap = make_snap(win_start, snap_ms, vec![
            (win_start,           dec!(63800.0)),
            (win_start + 30_000,  dec!(63820.0)),
            (win_start + 60_000,  dec!(63840.0)),
            (win_start + 90_000,  dec!(63860.0)),
            (win_start + 120_000, dec!(63880.0)),
            (win_start + 150_000, dec!(63900.0)),
            (snap_ms   -  1_000,  dec!(63920.0)),
        ]);
        let fv = compute_features(&snap);
        assert_eq!(fv.features.len(), FEATURE_COUNT,
            "features 长度应为 {FEATURE_COUNT}");
        assert_eq!(fv.feature_names.len(), FEATURE_COUNT,
            "feature_names 长度应为 {FEATURE_COUNT}");
    }

    // ── 特征名对应正确 ────────────────────────────────────────────────────────

    #[test]
    fn feature_names_match_registry() {
        let win_start = 1_786_401_600_000_i64;
        let snap_ms   = win_start + 190_000;
        let snap = make_snap(win_start, snap_ms, vec![
            (win_start, dec!(63900.0)),
            (snap_ms,   dec!(63920.0)),
        ]);
        let fv = compute_features(&snap);
        for (i, name) in fv.feature_names.iter().enumerate() {
            assert_eq!(name.as_str(), FEATURE_NAMES[i],
                "第 {i} 个特征名不匹配");
        }
    }

    // ── 上涨场景：关键特征方向正确 ───────────────────────────────────────────

    #[test]
    fn uptrend_features_direction() {
        let win_start = 1_786_401_600_000_i64;
        let snap_ms   = win_start + 190_000;
        let snap = make_snap(win_start, snap_ms, vec![
            (win_start,           dec!(63800.0)),
            (win_start + 30_000,  dec!(63820.0)),
            (win_start + 60_000,  dec!(63840.0)),
            (win_start + 90_000,  dec!(63860.0)),
            (win_start + 120_000, dec!(63880.0)),
            (win_start + 150_000, dec!(63900.0)),
            (snap_ms   -  1_000,  dec!(63920.0)),
        ]);
        let fv = compute_features(&snap);
        let idx = |name: &str| fv.feature_names.iter().position(|n| n == name).unwrap();

        // 上涨：ret_so_far > 0
        assert!(fv.features[idx("ret_so_far")] > 0.0,
            "上涨行情 ret_so_far 应为正");
        // 上涨：price_position 应接近 1（当前价接近最高）
        assert!(fv.features[idx("price_position")] > 0.9,
            "稳定上涨时 price_position 应接近 1.0");
        // 上涨：norm_dist > 0
        let nd = fv.features[idx("norm_dist")];
        assert!(!nd.is_nan(), "norm_dist 不应为 NAN");
        assert!(nd > 0.0, "上涨行情 norm_dist 应为正");
        // 上涨：全窗口斜率 > 0
        assert!(fv.features[idx("slope_all")] > 0.0,
            "上涨行情 slope_all 应为正");
    }

    // ── 单 tick 时大部分特征为 NAN，但向量长度仍固定 ─────────────────────────

    #[test]
    fn single_tick_nan_safety() {
        let win_start = 1_786_401_600_000_i64;
        let snap_ms   = win_start + 5_000;
        let snap = make_snap(win_start, snap_ms, vec![
            (win_start, dec!(63900.0)),
        ]);
        let fv = compute_features(&snap);
        // 向量长度仍必须固定
        assert_eq!(fv.features.len(), FEATURE_COUNT);
        // 标准差不足（n<2）→ norm_dist 必须是 NAN
        let idx_nd = fv.feature_names.iter().position(|n| n == "norm_dist").unwrap();
        assert!(fv.features[idx_nd].is_nan(), "单 tick 时 norm_dist 应为 NAN");
    }

    // ── 时间特征在 T-110s 时的值 ─────────────────────────────────────────────

    #[test]
    fn time_frac_at_pred_point() {
        let win_start = 1_786_401_600_000_i64;
        let snap_ms   = win_start + 190_000; // T-110s 预测时点
        let snap = make_snap(win_start, snap_ms, vec![
            (win_start, dec!(63900.0)),
            (snap_ms,   dec!(63910.0)),
        ]);
        let fv = compute_features(&snap);
        let idx = |name: &str| fv.feature_names.iter().position(|n| n == name).unwrap();

        let tf = fv.features[idx("time_frac")];
        // 190/300 ≈ 0.6333
        assert!((tf - 190.0/300.0).abs() < 1e-9, "time_frac 应为 190/300，got {tf}");
        let tfs = fv.features[idx("time_frac_sqrt")];
        assert!((tfs - (190.0/300.0_f64).sqrt()).abs() < 1e-9,
            "time_frac_sqrt 应为 √(190/300)，got {tfs}");
    }

    // ── 质量特征：tick 密度与 staleness ───────────────────────────────────────

    #[test]
    fn quality_features() {
        let win_start = 1_786_401_600_000_i64;
        let snap_ms   = win_start + 60_000; // 经过 60 秒
        // 放入 60 条 tick（每秒一条），最后一条在 snap_ms - 500ms
        let ticks: Vec<_> = (0..60)
            .map(|i| (win_start + i * 1000, dec!(63900.0)))
            .collect();
        let snap = make_snap(win_start, snap_ms, ticks);
        let fv = compute_features(&snap);
        let idx = |name: &str| fv.feature_names.iter().position(|n| n == name).unwrap();

        let density = fv.features[idx("tick_density")];
        // 60 ticks / 60s = 1.0
        assert!((density - 1.0).abs() < 0.01, "tick_density 应约为 1.0，got {density}");

        let stale = fv.features[idx("staleness_s")];
        // 最后一条在 win_start + 59000，距 snap_ms=win_start+60000 → 1s
        assert!((stale - 1.0).abs() < 0.01, "staleness_s 应约为 1.0s，got {stale}");
    }

    // ── E 组：交易所特征 ──────────────────────────────────────────────────────

    /// 现货高于 TWAP 时原始基差为正 —— TWAP 滞后，接下来会向上收敛
    #[test]
    fn spot_basis_positive_when_spot_leads_up() {
        let win_start = 1_786_401_600_000_i64;
        let snap_ms   = win_start + 190_000;
        // TWAP 停在 63900，现货三家都在 63950 附近
        let snap = make_snap_with_ex(
            win_start, snap_ms,
            vec![(win_start, dec!(63890.0)), (snap_ms, dec!(63900.0))],
            vec![
                bbo("binance",  snap_ms - 100, 63949.0, 1.0, 63951.0, 1.0),
                bbo("okx",      snap_ms - 200, 63948.0, 1.0, 63952.0, 1.0),
                bbo("coinbase", snap_ms - 300, 63950.0, 1.0, 63952.0, 1.0),
            ],
            vec![],
        );
        let basis = snap.spot_twap_basis().expect("应能算出基差");
        assert!(basis > 0.0, "现货高于 TWAP 时基差应为正，got {basis}");
        // (63950 - 63900) / 63900 ≈ 0.000782
        assert!((basis - 50.0/63900.0).abs() < 1e-5, "基差量级不对：{basis}");
    }

    /// 现货低于 TWAP 时原始基差为负
    #[test]
    fn spot_basis_negative_when_spot_leads_down() {
        let win_start = 1_786_401_600_000_i64;
        let snap_ms   = win_start + 190_000;
        let snap = make_snap_with_ex(
            win_start, snap_ms,
            vec![(win_start, dec!(63900.0)), (snap_ms, dec!(63900.0))],
            vec![
                bbo("binance", snap_ms - 100, 63849.0, 1.0, 63851.0, 1.0),
                bbo("okx",     snap_ms - 200, 63848.0, 1.0, 63852.0, 1.0),
            ],
            vec![],
        );
        assert!(snap.spot_twap_basis().unwrap() < 0.0, "现货低于 TWAP 时基差应为负");
    }

    /// 恒定基差（现货始终高 TWAP 固定比例）时，去均值后的偏离应约等于 0。
    ///
    /// 这正是实测中 +9bp 常数偏移的情形 —— 它不含方向信息，必须被消掉。
    #[test]
    fn constant_basis_yields_zero_deviation() {
        let win_start = 1_786_401_600_000_i64;
        let snap_ms   = win_start + 190_000;

        // TWAP 与现货同步上涨，两者始终保持 +9bp 的固定比例差
        // 步长 5 秒 × 38 步正好落在 snap_ms 上，保证最后一条报价不过期
        let mut twaps = Vec::new();
        let mut bbos  = Vec::new();
        for i in 0..=38 {
            let t  = win_start + i * 5_000;
            let px = 63_900.0 + i as f64 * 2.0;
            twaps.push((t, rust_decimal::Decimal::from_f64_retain(px).unwrap()));
            let mid = px * 1.0009;                    // 恒定 +9bp
            bbos.push(bbo("binance", t, mid - 0.5, 1.0, mid + 0.5, 1.0));
        }
        let snap = make_snap_with_ex(win_start, snap_ms, twaps, bbos, vec![]);

        let raw = snap.spot_twap_basis().expect("原始基差");
        assert!((raw - 0.0009).abs() < 1e-5, "原始基差应约 +9bp，got {raw}");

        let dev = snap.spot_basis_dev().expect("去均值基差");
        assert!(dev.abs() < 1e-6, "恒定基差去均值后应约 0，got {dev}");
    }

    /// 现货在窗口末尾相对常态突然抬升时，去均值基差为正 —— 这才是信号。
    #[test]
    fn basis_deviation_captures_late_spot_move() {
        let win_start = 1_786_401_600_000_i64;
        let snap_ms   = win_start + 190_000;

        let mut twaps = Vec::new();
        let mut bbos  = Vec::new();
        for i in 0..=38 {
            let t = win_start + i * 5_000;
            twaps.push((t, dec!(63900.0)));           // TWAP 纹丝不动（滞后平均）
            // 前 30 条稳定在 +9bp，最后几条现货抬到 +20bp
            let bp = if i < 30 { 0.0009 } else { 0.0020 };
            let mid = 63_900.0 * (1.0 + bp);
            bbos.push(bbo("binance", t, mid - 0.5, 1.0, mid + 0.5, 1.0));
        }
        let snap = make_snap_with_ex(win_start, snap_ms, twaps, bbos, vec![]);

        let dev = snap.spot_basis_dev().expect("去均值基差");
        assert!(dev > 0.0, "现货末段上冲时偏离应为正，got {dev}");

        let z = snap.spot_basis_dev_z().expect("z 化偏离");
        assert!(z > 0.5, "偏离应显著（z>0.5），got {z}");
    }

    /// 过期报价不参与共识：kraken 十几秒不更新时，不该被当成"分歧"
    #[test]
    fn stale_quotes_are_excluded() {
        use crate::snapshot::BBO_STALE_MS;
        let win_start = 1_786_401_600_000_i64;
        let snap_ms   = win_start + 190_000;
        let snap = make_snap_with_ex(
            win_start, snap_ms,
            vec![(win_start, dec!(63900.0))],
            vec![
                // kraken 的报价来自 12 秒前，且偏离 60 美元
                bbo("kraken",  snap_ms - 12_000, 63839.0, 1.0, 63841.0, 1.0),
                bbo("binance", snap_ms - 100,    63899.0, 1.0, 63901.0, 1.0),
                bbo("okx",     snap_ms - 100,    63900.0, 1.0, 63902.0, 1.0),
            ],
            vec![],
        );
        assert!(BBO_STALE_MS < 12_000, "阈值必须能挡住 12 秒的陈旧报价");
        assert_eq!(snap.active_exchange_count(), 2, "kraken 应被剔除");

        let disp = snap.cross_exchange_dispersion().expect("分歧度");
        // binance mid=63900，okx mid=63901，差 1 美元 → 约 1.6bp 以内
        assert!(disp < 5e-5, "剔除陈旧报价后分歧度应很小，got {disp}");
    }

    /// consensus_mid 用中位数：单家抽风不该带偏信号
    #[test]
    fn consensus_mid_resists_one_bad_exchange() {
        let win_start = 1_786_401_600_000_i64;
        let snap_ms   = win_start + 190_000;
        let snap = make_snap_with_ex(
            win_start, snap_ms,
            vec![(win_start, dec!(63900.0))],
            vec![
                bbo("binance",  snap_ms - 100, 63899.0, 1.0, 63901.0, 1.0),
                bbo("okx",      snap_ms - 100, 63900.0, 1.0, 63902.0, 1.0),
                bbo("bybit",    snap_ms - 100, 63898.0, 1.0, 63902.0, 1.0),
                // kraken 报价卡在一小时前的价位，偏离 10%
                bbo("kraken",   snap_ms - 100, 57000.0, 1.0, 57010.0, 1.0),
            ],
            vec![],
        );
        let mid = snap.consensus_mid().expect("应有共识中间价");
        // 中位数应落在正常三家附近，不被 kraken 拖走
        assert!((mid - 63900.0).abs() < 50.0,
            "中位数应抵抗单家异常报价，got {mid}");
    }

    /// 每家交易所只取最新一条 BBO
    #[test]
    fn latest_bbo_dedups_per_exchange() {
        let win_start = 1_786_401_600_000_i64;
        let snap_ms   = win_start + 190_000;
        let snap = make_snap_with_ex(
            win_start, snap_ms,
            vec![(win_start, dec!(63900.0))],
            vec![
                bbo("binance", snap_ms - 5000, 63800.0, 1.0, 63801.0, 1.0), // 旧
                bbo("binance", snap_ms - 100,  63900.0, 1.0, 63901.0, 1.0), // 新
                bbo("okx",     snap_ms - 200,  63899.0, 1.0, 63902.0, 1.0),
            ],
            vec![],
        );
        let latest = snap.latest_bbo_per_exchange();
        assert_eq!(latest.len(), 2, "两家交易所应各取一条");
        let binance = latest.iter().find(|t| t.exchange == "binance").unwrap();
        assert_eq!(binance.recv_ms, snap_ms - 100, "binance 应取最新那条");
    }

    /// 成交流失衡：全买 → +1，全卖 → -1，均衡 → 0
    #[test]
    fn trade_flow_imbalance_range() {
        let win_start = 1_786_401_600_000_i64;
        let snap_ms   = win_start + 190_000;

        let all_buy = make_snap_with_ex(
            win_start, snap_ms, vec![(win_start, dec!(63900.0))], vec![],
            vec![
                trade("binance", snap_ms - 5_000, 63900.0, 1.0, true),
                trade("binance", snap_ms - 3_000, 63900.0, 2.0, true),
            ],
        );
        assert_eq!(all_buy.trade_flow_imbalance(30_000), Some(1.0), "全买应为 +1");

        let all_sell = make_snap_with_ex(
            win_start, snap_ms, vec![(win_start, dec!(63900.0))], vec![],
            vec![trade("binance", snap_ms - 5_000, 63900.0, 1.0, false)],
        );
        assert_eq!(all_sell.trade_flow_imbalance(30_000), Some(-1.0), "全卖应为 -1");

        let balanced = make_snap_with_ex(
            win_start, snap_ms, vec![(win_start, dec!(63900.0))], vec![],
            vec![
                trade("binance", snap_ms - 5_000, 63900.0, 1.0, true),
                trade("binance", snap_ms - 4_000, 63900.0, 1.0, false),
            ],
        );
        assert_eq!(balanced.trade_flow_imbalance(30_000), Some(0.0), "买卖相等应为 0");
    }

    /// 成交流窗口边界：窗口外的成交不能计入
    #[test]
    fn trade_flow_respects_window() {
        let win_start = 1_786_401_600_000_i64;
        let snap_ms   = win_start + 190_000;
        let snap = make_snap_with_ex(
            win_start, snap_ms, vec![(win_start, dec!(63900.0))], vec![],
            vec![
                // 40 秒前的卖单 —— 在 30s 窗口之外
                trade("binance", snap_ms - 40_000, 63900.0, 100.0, false),
                // 10 秒前的买单 —— 在窗口内
                trade("binance", snap_ms - 10_000, 63900.0, 1.0, true),
            ],
        );
        // 30s 窗口只应看到那笔买单
        assert_eq!(snap.trade_flow_imbalance(30_000), Some(1.0),
            "30s 窗口不应包含 40 秒前的成交");
        // 60s 窗口两笔都看到，卖量占压倒多数
        let imb60 = snap.trade_flow_imbalance(60_000).unwrap();
        assert!(imb60 < -0.9, "60s 窗口应被大额卖单主导，got {imb60}");
    }

    /// 盘口失衡：买盘厚 → >0.5
    #[test]
    fn book_imbalance_reflects_depth() {
        let win_start = 1_786_401_600_000_i64;
        let snap_ms   = win_start + 190_000;
        let snap = make_snap_with_ex(
            win_start, snap_ms, vec![(win_start, dec!(63900.0))],
            vec![
                // bid_qty 3 : ask_qty 1 → imbalance 0.75
                bbo("binance", snap_ms - 100, 63899.0, 3.0, 63901.0, 1.0),
                bbo("okx",     snap_ms - 100, 63899.0, 3.0, 63901.0, 1.0),
            ],
            vec![],
        );
        let imb = snap.book_imbalance().expect("应有盘口失衡");
        assert!((imb - 0.75).abs() < 1e-9, "买盘 3:1 时应为 0.75，got {imb}");
    }

    /// 没有交易所数据时，E 组特征为 NAN，但向量长度不变
    #[test]
    fn missing_exchange_data_is_nan_safe() {
        let win_start = 1_786_401_600_000_i64;
        let snap_ms   = win_start + 190_000;
        let snap = make_snap(win_start, snap_ms, vec![
            (win_start, dec!(63900.0)),
            (snap_ms,   dec!(63910.0)),
        ]);
        let fv = compute_features(&snap);
        assert_eq!(fv.features.len(), FEATURE_COUNT, "向量长度必须固定");

        let idx = |n: &str| fv.feature_names.iter().position(|x| x == n).unwrap();
        for name in ["spot_basis_dev", "spot_basis_dev_z", "flow_imb_30s", "book_imb", "rel_spread"] {
            assert!(fv.features[idx(name)].is_nan(),
                "无交易所数据时 {name} 应为 NAN");
        }
        // 计数类特征应为 0 而非 NAN
        assert_eq!(fv.features[idx("n_exchanges")], 0.0);
        assert_eq!(fv.features[idx("log_trades_30s")], 0.0, "ln(1+0) = 0");
    }
}

// ── 特征名称注册表 ────────────────────────────────────────────────────────────

/// 特征总数，与下面的 FEATURE_NAMES 必须保持同步
pub const FEATURE_COUNT: usize = 26;

/// 特征名称，与 compute_features 输出顺序严格对应
pub const FEATURE_NAMES: [&str; FEATURE_COUNT] = [
    // A. 位置类（TWAP）
    "ret_so_far",       // A1: 窗口内总收益率
    "price_position",   // A2: 价格在最高最低之间的位置 [0,1]
    "dist_from_hi",     // A3: 距最高价的相对距离
    "dist_from_lo",     // A4: 距最低价的相对距离
    // B. 动量类（TWAP）
    "ret_30s",          // B1: 最近 30s 收益率
    "ret_60s",          // B2: 最近 60s 收益率
    "ret_90s",          // B3: 最近 90s 收益率
    "slope_all",        // B4: 全窗口线性斜率（相对，/秒）
    "slope_recent",     // B5: 最近 60s 线性斜率（相对，/秒）
    // C. 结构类（TWAP）
    "norm_dist",        // C1: 归一化距离
    "norm_dist_sq",     // C2: 归一化距离的平方
    "time_frac",        // C3: 时间分数 [0,1]
    "time_frac_sqrt",   // C4: √时间分数
    "rel_vol",          // C5: 相对波动率
    // D. 质量类（TWAP）
    "tick_density",     // D1: tick/秒
    "staleness_s",      // D2: 最近 tick 距 snap_ms 的延迟（秒）
    // E. 交易所类
    "spot_basis_dev",   // E1: 基差相对窗口均值的偏离（已去掉 ~+9bp 常数偏移）
    "spot_basis_dev_z", // E2: 同一偏离 / 其窗口内标准差
    "flow_imb_30s",     // E3: 最近 30s 成交流失衡 [-1,1]
    "flow_imb_60s",     // E4: 最近 60s 成交流失衡
    "flow_imb_win",     // E5: 全窗口成交流失衡
    "book_imb",         // E6: 盘口失衡（bid 占比均值）
    "rel_spread",       // E7: 平均相对价差
    "xex_dispersion",   // E8: 跨交易所价格分歧度
    "log_trades_30s",   // E9: ln(1 + 最近 30s 成交笔数)
    "n_exchanges",      // E10: 有报价的交易所家数
];
