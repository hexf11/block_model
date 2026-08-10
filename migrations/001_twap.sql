-- Polymarket RTDS Chainlink TWAP feed
CREATE DATABASE IF NOT EXISTS pm;

CREATE TABLE IF NOT EXISTS pm.twap
(
    symbol      LowCardinality(String),
    window_s    UInt16,
    obs_ms      UInt64,                       -- Chainlink 观测时间
    pub_ms      UInt64,                       -- RTDS 发布时间
    recv_ms     UInt64,                       -- 本地接收时间
    value_e18   String,                       -- Chainlink 原始 E18 整数
    price       Decimal128(18),               -- value_e18 / 1e18，i128 wire 类型

    obs_ts      DateTime64(3) MATERIALIZED toDateTime64(obs_ms / 1000.0, 3),
    pub_lag_ms  Int64         MATERIALIZED toInt64(pub_ms)  - toInt64(obs_ms),
    recv_lag_ms Int64         MATERIALIZED toInt64(recv_ms) - toInt64(pub_ms)
)
ENGINE = ReplacingMergeTree(recv_ms)
PARTITION BY toYYYYMMDD(obs_ts)
ORDER BY (symbol, window_s, obs_ms);

-- 所有采集器的连接生命周期事件
CREATE TABLE IF NOT EXISTS pm.conn_events
(
    ts_ms   UInt64,
    source  LowCardinality(String),           -- rtds / binance / okx / ...
    event   LowCardinality(String),           -- connected / disconnected / stale_reconnect / subscribe_error
    detail  String,

    ts      DateTime64(3) MATERIALIZED toDateTime64(ts_ms / 1000.0, 3)
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(ts)
ORDER BY ts_ms;

-- 每个 symbol 的最新 TWAP
CREATE VIEW IF NOT EXISTS pm.twap_latest AS
SELECT
    symbol,
    window_s,
    argMax(price,  obs_ms) AS price,
    argMax(obs_ms, obs_ms) AS obs_ms,
    argMax(recv_ms, obs_ms) AS recv_ms
FROM pm.twap
GROUP BY symbol, window_s;
