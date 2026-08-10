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
    use crate::snapshot::{TwapTick, WindowSnapshot};
    use rust_decimal_macros::dec;

    fn make_snap(win_start_ms: i64, snap_ms: i64, ticks: Vec<(i64, rust_decimal::Decimal)>) -> WindowSnapshot {
        WindowSnapshot {
            snap_ms,
            symbol: "btc/usd".to_string(),
            win_start_ms,
            twap_ticks: ticks.into_iter().map(|(obs_ms, price)| TwapTick { obs_ms, price }).collect(),
        }
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
}

// ── 特征名称注册表 ────────────────────────────────────────────────────────────

/// 特征总数，与下面的 FEATURE_NAMES 必须保持同步
pub const FEATURE_COUNT: usize = 16;

/// 特征名称，与 compute_features 输出顺序严格对应
pub const FEATURE_NAMES: [&str; FEATURE_COUNT] = [
    // A. 位置类
    "ret_so_far",       // A1: 窗口内总收益率
    "price_position",   // A2: 价格在最高最低之间的位置 [0,1]
    "dist_from_hi",     // A3: 距最高价的相对距离
    "dist_from_lo",     // A4: 距最低价的相对距离
    // B. 动量类
    "ret_30s",          // B1: 最近 30s 收益率
    "ret_60s",          // B2: 最近 60s 收益率
    "ret_90s",          // B3: 最近 90s 收益率
    "slope_all",        // B4: 全窗口线性斜率（相对，/秒）
    "slope_recent",     // B5: 最近 60s 线性斜率（相对，/秒）
    // C. 结构类
    "norm_dist",        // C1: 归一化距离
    "norm_dist_sq",     // C2: 归一化距离的平方
    "time_frac",        // C3: 时间分数 [0,1]
    "time_frac_sqrt",   // C4: √时间分数
    "rel_vol",          // C5: 相对波动率
    // D. 质量类（携带在向量末尾，训练时按名称决定是否用作模型输入）
    "tick_density",     // D1: tick/秒
    "staleness_s",      // D2: 最近 tick 距 snap_ms 的延迟（秒）
];
