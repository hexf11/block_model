-- Polymarket 5 分钟 UP/DOWN 窗口切分与标签
--
-- 一个窗口 = 一个训练样本。这里把 pm.twap 的逐秒 tick 聚合成窗口级别的
-- 开盘价 / 收盘价 / 标签，并附带数据质量标记，让下游可以按需过滤。
--
-- ⚠️ 待核对的假设：
--   开盘价这里取"窗口内第一条 TWAP 观测"，收盘价取"最后一条"。Polymarket
--   实际结算用的是边界时刻的 Chainlink 轮次，可能与此有细微差别（比如用
--   窗口开始前最后一条，而非开始后第一条）。必须拿真实市场的结算结果
--   反向核对过再用于训练 —— 错一个 tick 就会让标签系统性偏移。

-- ── 窗口级别聚合 ────────────────────────────────────────────────────────────
--
-- 时间轴（win_start = W，窗口长 300s）：
--
--   W          W+190s              W+300s
--   │            │                    │
--   开盘         预测时点 T-110s      收盘
--   open_px      px_at_pred           close_px
--                └─ 只允许看这之前的数据 ─┘
--
CREATE VIEW IF NOT EXISTS pm.market_windows AS
SELECT
    symbol,
    win_start_ms,
    win_start_ms + 300000 AS win_end_ms,
    -- 预测时点：收盘前 110 秒
    win_start_ms + 190000 AS pred_ms,
    toDateTime(intDiv(win_start_ms, 1000)) AS win_start,

    open_px,
    close_px,
    -- px_at_pred 只由 obs_ms <= pred_ms 的 tick 得出，这是防未来信息泄漏的
    -- 硬边界；任何特征都不得越过这条线。
    px_at_pred,

    -- 标签：收盘 > 开盘 记为 UP。相等按 DOWN 处理（与多数二元市场的
    -- "严格大于才算 UP" 规则一致，但同样需要拿真实结算核对）。
    if(close_px > open_px, 1, 0) AS label_up,

    -- 相对涨跌幅，用于识别接近平局的窗口
    toFloat64(close_px - open_px) / toFloat64(open_px) AS rel_move,

    -- ── 数据质量 ──────────────────────────────────────────────────────────
    n_ticks,
    max_gap_ms,                                   -- 窗口内最大 tick 间隔
    toInt64(last_obs_ms) - toInt64(first_obs_ms) AS span_ms,
    -- 预测时点的数据新鲜度：pred_ms 与该点之前最后一条观测的时间差。
    -- 过大说明预测时刻手上只有过期价格，这种样本必须丢弃。
    if(pred_obs_ms = 0, 999999, toInt64(win_start_ms + 190000) - toInt64(pred_obs_ms)) AS pred_stale_ms,

    -- 综合可用性判定。阈值说明：
    --   n_ticks      —— 满窗约 300 条（1/s）；RTDS 有轻微抖动，实测正常窗口
    --                   在 280~290，取 240（80%）作为下限
    --   max_gap_ms   —— 单次间隔超 5s 说明有断线，窗口内价格路径不可信
    --   pred_stale_ms—— 预测时点价格超过 5s 未更新，特征失去意义
    multiIf(
        n_ticks < 240,        'broken',
        max_gap_ms > 5000,    'gap',
        pred_stale_ms > 5000, 'stale_at_pred',
        'ok'
    ) AS quality,

    -- 接近平局标记。这类窗口的标签本质是噪音（实测有过 BTC 一个窗口
    -- 只动 0.6 USD、约 1e-5 的情况），留给 research 侧决定是否剔除，
    -- 不在这里过滤。
    abs(rel_move) < 0.00002 AS near_tie

FROM (
    SELECT
        symbol,
        intDiv(obs_ms, 300000) * 300000 AS win_start_ms,

        argMin(price, obs_ms) AS open_px,
        argMax(price, obs_ms) AS close_px,

        -- 预测时点及其之前的最后一条观测
        argMaxIf(price,  obs_ms, obs_ms <= intDiv(obs_ms, 300000) * 300000 + 190000) AS px_at_pred,
        maxIf(obs_ms,            obs_ms <= intDiv(obs_ms, 300000) * 300000 + 190000) AS pred_obs_ms,

        count()      AS n_ticks,
        min(obs_ms)  AS first_obs_ms,
        max(obs_ms)  AS last_obs_ms,

        -- 先排序再求差，保证差值非负；转 Int64 避免 UInt64 下溢
        arrayMax(arrayDifference(arraySort(groupArray(toInt64(obs_ms))))) AS max_gap_ms

    FROM pm.twap
    WHERE window_s = 30
    GROUP BY symbol, win_start_ms
);

-- ── 只含可训练样本的便捷 view ────────────────────────────────────────────────
--
-- 同时排除首尾不完整的窗口：采集器启动/停止那一刻的窗口天然残缺，
-- quality 已能识别，这里不再重复判断。
CREATE VIEW IF NOT EXISTS pm.training_windows AS
SELECT *
FROM pm.market_windows
WHERE quality = 'ok';
