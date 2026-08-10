-- 交易所行情数据：BBO / 成交 / L2 盘口

CREATE TABLE IF NOT EXISTS pm.ex_bbo
(
    exchange LowCardinality(String),
    symbol   LowCardinality(String),
    ts_ex    UInt64,
    recv_ms  UInt64,
    bid      String,
    bid_qty  String,
    ask      String,
    ask_qty  String,

    ts       DateTime64(3) MATERIALIZED toDateTime64(ts_ex / 1000.0, 3)
)
ENGINE = MergeTree
PARTITION BY (exchange, toYYYYMMDD(ts))
ORDER BY (exchange, symbol, ts_ex)
TTL toDateTime(ts) + INTERVAL 90 DAY;

CREATE TABLE IF NOT EXISTS pm.ex_trades
(
    exchange LowCardinality(String),
    symbol   LowCardinality(String),
    ts_ex    UInt64,
    recv_ms  UInt64,
    price    String,
    qty      String,
    side     LowCardinality(String),          -- buy / sell（taker 方向）
    trade_id String,

    ts       DateTime64(3) MATERIALIZED toDateTime64(ts_ex / 1000.0, 3)
)
ENGINE = ReplacingMergeTree(recv_ms)
PARTITION BY (exchange, toYYYYMMDD(ts))
ORDER BY (exchange, symbol, ts_ex, trade_id)
TTL toDateTime(ts) + INTERVAL 90 DAY;

CREATE TABLE IF NOT EXISTS pm.ex_book
(
    exchange    LowCardinality(String),
    symbol      LowCardinality(String),
    ts_ex       UInt64,
    recv_ms     UInt64,
    bids_json   String,                       -- [[price, qty], ...]
    asks_json   String,
    is_snapshot UInt8,
    seq         UInt64,

    ts          DateTime64(3) MATERIALIZED toDateTime64(ts_ex / 1000.0, 3)
)
ENGINE = MergeTree
PARTITION BY (exchange, toYYYYMMDD(ts))
-- 单靠 ts_ex 不能保证唯一性：Binance 和 Bybit 都会在同一毫秒内
-- 发出多条深度增量。增量乱序重放会破坏重建出来的盘口，
-- 因此排序键必须在毫秒内确定一个因果顺序。
--
-- recv_ms 排在 seq 之前，是因为它对每家交易所都具备因果性，
-- 而 seq 只在 Binance/OKX/Bybit/Coinbase 上才是真正的序列号——
-- Kraken 在该字段里放的是盘口校验和，虽然确定但并无序。
ORDER BY (exchange, symbol, ts_ex, recv_ms, seq)
TTL toDateTime(ts) + INTERVAL 30 DAY;
