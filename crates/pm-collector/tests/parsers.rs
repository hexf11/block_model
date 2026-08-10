//! Parser tests against frames shaped like the real venue payloads.
//!
//! These lock down the fields that are easy to get subtly wrong and expensive
//! to notice later: taker side inference, timestamp source (venue vs local),
//! snapshot-vs-delta classification, and symbol normalisation.

use pm_collector::exchanges::{
    binance::BinanceCollector, bybit::BybitCollector, coinbase::CoinbaseCollector,
    kraken::KrakenCollector, okx::OkxCollector, ExchangeSpec, ParsedRecord,
};
use pm_features::types::TradeSide;
use rust_decimal::Decimal;
use std::str::FromStr;

const RECV: i64 = 1_754_900_000_000;

fn d(s: &str) -> Decimal { Decimal::from_str(s).unwrap() }

// ── extraction helpers ───────────────────────────────────────────────────────

fn one_bbo(recs: Vec<ParsedRecord>) -> pm_features::types::BboRecord {
    let mut it = recs.into_iter().filter_map(|r| match r {
        ParsedRecord::Bbo(b) => Some(b),
        _ => None,
    });
    let first = it.next().expect("expected exactly one BBO record, got none");
    assert!(it.next().is_none(), "expected exactly one BBO record, got more");
    first
}

fn trades(recs: Vec<ParsedRecord>) -> Vec<pm_features::types::TradeRecord> {
    recs.into_iter().filter_map(|r| match r {
        ParsedRecord::Trade(t) => Some(t),
        _ => None,
    }).collect()
}

fn one_book(recs: Vec<ParsedRecord>) -> pm_features::types::BookRecord {
    let mut it = recs.into_iter().filter_map(|r| match r {
        ParsedRecord::Book(b) => Some(b),
        _ => None,
    });
    let first = it.next().expect("expected exactly one book record, got none");
    assert!(it.next().is_none(), "expected exactly one book record, got more");
    first
}

// ── Binance ──────────────────────────────────────────────────────────────────

#[test]
fn binance_book_ticker() {
    let raw = r#"{"stream":"btcusdt@bookTicker","data":{
        "u":400900217,"s":"BTCUSDT",
        "b":"25.35190000","B":"31.21000000",
        "a":"25.36520000","A":"40.66000000"}}"#;
    let b = one_bbo(BinanceCollector::new().parse(raw, RECV));
    assert_eq!(b.symbol, "btc/usd");
    assert_eq!(b.bid, d("25.35190000"));
    assert_eq!(b.ask_qty, d("40.66000000"));
    // bookTicker carries no event time, so ts_ex must fall back to recv_ms
    assert_eq!(b.ts_ex, RECV);
}

#[test]
fn binance_agg_trade_maker_buyer_means_taker_sell() {
    let raw = r#"{"stream":"ethusdt@aggTrade","data":{
        "e":"aggTrade","E":1754899999999,"s":"ETHUSDT","a":12345,
        "p":"3011.50","q":"0.480","f":100,"l":105,
        "T":1754899999888,"m":true,"M":true}}"#;
    let t = trades(BinanceCollector::new().parse(raw, RECV));
    assert_eq!(t.len(), 1);
    assert_eq!(t[0].symbol, "eth/usd");
    // m=true → maker was the buyer → the aggressor sold
    assert_eq!(t[0].side, TradeSide::Sell);
    assert_eq!(t[0].ts_ex, 1_754_899_999_888, "must use venue T, not recv_ms");
    assert_eq!(t[0].trade_id, "12345");
}

#[test]
fn binance_agg_trade_maker_seller_means_taker_buy() {
    let raw = r#"{"stream":"solusdt@aggTrade","data":{
        "s":"SOLUSDT","a":9,"p":"180.1","q":"2.5","T":1754899999000,"m":false}}"#;
    let t = trades(BinanceCollector::new().parse(raw, RECV));
    assert_eq!(t[0].side, TradeSide::Buy);
}

#[test]
fn binance_depth_is_never_snapshot() {
    let raw = r#"{"stream":"btcusdt@depth@100ms","data":{
        "e":"depthUpdate","E":1754899999999,"T":1754899999900,"s":"BTCUSDT",
        "U":157,"u":160,
        "b":[["0.0024","10"],["0.0023","5"]],
        "a":[["0.0026","100"]]}}"#;
    let b = one_book(BinanceCollector::new().parse(raw, RECV));
    assert!(!b.is_snapshot, "the diff-depth stream only ever sends deltas");
    assert_eq!(b.seq, Some(160));
    assert_eq!(b.bids.len(), 2);
    assert_eq!(b.asks[0], [d("0.0026"), d("100")]);
}

#[test]
fn binance_ignores_unknown_symbol_and_subscribe_ack() {
    let c = BinanceCollector::new();
    assert!(c.parse(r#"{"result":null,"id":1}"#, RECV).is_empty());
    assert!(c.parse(r#"{"stream":"adausdt@bookTicker","data":{"b":"1","a":"2"}}"#, RECV).is_empty());
}

// ── OKX ──────────────────────────────────────────────────────────────────────

#[test]
fn okx_bbo_tbt() {
    let raw = r#"{"arg":{"channel":"bbo-tbt","instId":"BTC-USDT"},"data":[{
        "asks":[["112000.1","0.5","0","2"]],
        "bids":[["111999.9","1.25","0","3"]],
        "ts":"1754899999123","seqId":998}]}"#;
    let b = one_bbo(OkxCollector::new().parse(raw, RECV));
    assert_eq!(b.symbol, "btc/usd");
    assert_eq!(b.bid, d("111999.9"));
    assert_eq!(b.bid_qty, d("1.25"));
    assert_eq!(b.ask, d("112000.1"));
    // OKX sends ts as a stringified integer
    assert_eq!(b.ts_ex, 1_754_899_999_123);
}

#[test]
fn okx_trades_batch() {
    let raw = r#"{"arg":{"channel":"trades","instId":"DOGE-USDT"},"data":[
        {"tradeId":"1","px":"0.2201","sz":"1000","side":"buy","ts":"1754899999001"},
        {"tradeId":"2","px":"0.2200","sz":"500","side":"sell","ts":"1754899999002"}]}"#;
    let t = trades(OkxCollector::new().parse(raw, RECV));
    assert_eq!(t.len(), 2, "every element of data must yield a trade");
    assert_eq!(t[0].symbol, "doge/usd");
    assert_eq!(t[0].side, TradeSide::Buy);
    assert_eq!(t[1].side, TradeSide::Sell);
    assert_eq!(t[1].ts_ex, 1_754_899_999_002);
}

#[test]
fn okx_books5_is_always_snapshot() {
    let raw = r#"{"arg":{"channel":"books5","instId":"ETH-USDT"},"data":[{
        "asks":[["3011.6","2","0","1"]],"bids":[["3011.4","3","0","1"]],
        "ts":"1754899999500","seqId":4242}]}"#;
    let b = one_book(OkxCollector::new().parse(raw, RECV));
    assert!(b.is_snapshot, "books5 pushes the full top-5 book each time");
    assert_eq!(b.seq, Some(4242));
}

#[test]
fn okx_ignores_control_frames() {
    let c = OkxCollector::new();
    assert!(c.parse("pong", RECV).is_empty());
    assert!(c.parse(r#"{"event":"subscribe","arg":{"channel":"trades","instId":"BTC-USDT"}}"#, RECV).is_empty());
    assert!(c.parse(r#"{"event":"error","code":"60012","msg":"bad request"}"#, RECV).is_empty());
}

// ── Bybit ────────────────────────────────────────────────────────────────────

#[test]
fn bybit_orderbook_1_as_bbo() {
    let raw = r#"{"topic":"orderbook.1.BTCUSDT","type":"delta","ts":1754899999777,
        "data":{"s":"BTCUSDT","b":[["111999.5","2.1"]],"a":[["112000.5","0.9"]],"u":18521,"seq":77}}"#;
    let b = one_bbo(BybitCollector::new().parse(raw, RECV));
    assert_eq!(b.symbol, "btc/usd");
    assert_eq!(b.bid, d("111999.5"));
    assert_eq!(b.ask_qty, d("0.9"));
    assert_eq!(b.ts_ex, 1_754_899_999_777);
}

#[test]
fn bybit_one_sided_depth1_delta_is_dropped() {
    // A depth-1 delta that only touches one side would produce a zero on the
    // other; emitting that row would poison spread and mid features.
    let raw = r#"{"topic":"orderbook.1.ETHUSDT","type":"delta","ts":1754899999777,
        "data":{"s":"ETHUSDT","b":[["3011.4","1.0"]],"a":[],"u":2,"seq":3}}"#;
    assert!(BybitCollector::new().parse(raw, RECV).is_empty());
}

#[test]
fn bybit_public_trade() {
    let raw = r#"{"topic":"publicTrade.SOLUSDT","type":"snapshot","ts":1754899999000,
        "data":[{"T":1754899998950,"s":"SOLUSDT","S":"Sell","v":"12.5","p":"180.25","i":"abc-1"}]}"#;
    let t = trades(BybitCollector::new().parse(raw, RECV));
    assert_eq!(t.len(), 1);
    assert_eq!(t[0].symbol, "sol/usd");
    assert_eq!(t[0].side, TradeSide::Sell, "Bybit capitalises the side field");
    assert_eq!(t[0].ts_ex, 1_754_899_998_950, "per-trade T wins over envelope ts");
    assert_eq!(t[0].trade_id, "abc-1");
}

#[test]
fn bybit_orderbook_50_snapshot_flag() {
    let base = |ty: &str, u: u64| format!(
        r#"{{"topic":"orderbook.50.XRPUSDT","type":"{ty}","ts":1754899999000,
            "data":{{"s":"XRPUSDT","b":[["2.50","100"]],"a":[["2.51","200"]],"u":{u},"seq":9}}}}"#);
    let c = BybitCollector::new();

    assert!(one_book(c.parse(&base("snapshot", 500), RECV)).is_snapshot);
    assert!(!one_book(c.parse(&base("delta", 500), RECV)).is_snapshot);
    // u == 1 means Bybit restarted the service and reset the book, so the frame
    // must be applied as a snapshot even though it is labelled a delta.
    assert!(one_book(c.parse(&base("delta", 1), RECV)).is_snapshot,
        "u==1 signals a book reset");
}

#[test]
fn bybit_ignores_ack_frames() {
    let c = BybitCollector::new();
    assert!(c.parse(r#"{"success":true,"op":"subscribe","conn_id":"x"}"#, RECV).is_empty());
    assert!(c.parse(r#"{"op":"pong","args":["1754899999000"]}"#, RECV).is_empty());
}

// ── Coinbase ─────────────────────────────────────────────────────────────────

#[test]
fn coinbase_ticker() {
    let raw = r#"{"channel":"ticker","sequence_num":12,"timestamp":"2026-08-11T09:33:20.123456Z",
        "events":[{"type":"update","tickers":[{
            "type":"ticker","product_id":"BTC-USD",
            "price":"112000","best_bid":"111999.01","best_bid_quantity":"0.5",
            "best_ask":"112000.99","best_ask_quantity":"1.5"}]}]}"#;
    let b = one_bbo(CoinbaseCollector::new().parse(raw, RECV));
    assert_eq!(b.symbol, "btc/usd");
    assert_eq!(b.bid, d("111999.01"));
    assert_eq!(b.ask_qty, d("1.5"));
    // RFC-3339 timestamp must be converted to epoch ms, not left as recv_ms
    assert_ne!(b.ts_ex, RECV);
    assert_eq!(b.ts_ex, 1_786_440_800_123);
}

#[test]
fn coinbase_market_trades() {
    let raw = r#"{"channel":"market_trades","sequence_num":13,"timestamp":"2026-08-11T09:33:20.000Z",
        "events":[{"type":"update","trades":[
            {"trade_id":"t1","product_id":"ETH-USD","price":"3011.5","size":"0.25",
             "side":"BUY","time":"2026-08-11T09:33:19.500Z"}]}]}"#;
    let t = trades(CoinbaseCollector::new().parse(raw, RECV));
    assert_eq!(t.len(), 1);
    assert_eq!(t[0].symbol, "eth/usd");
    assert_eq!(t[0].side, TradeSide::Buy, "side arrives uppercase");
    assert_eq!(t[0].ts_ex, 1_786_440_799_500, "per-trade time, not envelope");
}

#[test]
fn coinbase_l2_offer_side_maps_to_asks() {
    // Coinbase spells the ask side "offer"; mapping it to bids would invert the book.
    let raw = r#"{"channel":"l2_data","sequence_num":99,"timestamp":"2026-08-11T09:33:20.000Z",
        "events":[{"type":"snapshot","product_id":"SOL-USD","updates":[
            {"side":"bid","event_time":"2026-08-11T09:33:20.000Z","price_level":"180.10","new_quantity":"5"},
            {"side":"offer","event_time":"2026-08-11T09:33:20.000Z","price_level":"180.20","new_quantity":"7"}]}]}"#;
    let b = one_book(CoinbaseCollector::new().parse(raw, RECV));
    assert_eq!(b.symbol, "sol/usd");
    assert!(b.is_snapshot);
    assert_eq!(b.bids, vec![[d("180.10"), d("5")]]);
    assert_eq!(b.asks, vec![[d("180.20"), d("7")]]);
    assert_eq!(b.seq, Some(99));
}

#[test]
fn coinbase_l2_delta_not_marked_snapshot() {
    let raw = r#"{"channel":"l2_data","sequence_num":100,"timestamp":"2026-08-11T09:33:21.000Z",
        "events":[{"type":"update","product_id":"BTC-USD","updates":[
            {"side":"bid","price_level":"111000","new_quantity":"0"}]}]}"#;
    let b = one_book(CoinbaseCollector::new().parse(raw, RECV));
    assert!(!b.is_snapshot);
    // quantity 0 is a level removal and must survive parsing so the book can
    // apply the delete
    assert_eq!(b.bids, vec![[d("111000"), Decimal::ZERO]]);
}

#[test]
fn coinbase_l2_data_and_level2_both_accepted() {
    // Subscribing to "level2" yields frames tagged "l2_data"; accept both.
    let c = CoinbaseCollector::new();
    let mk = |ch: &str| format!(
        r#"{{"channel":"{ch}","sequence_num":1,"timestamp":"2026-08-11T09:33:20.000Z",
            "events":[{{"type":"snapshot","product_id":"XRP-USD","updates":[
                {{"side":"bid","price_level":"2.5","new_quantity":"10"}}]}}]}}"#);
    assert_eq!(c.parse(&mk("level2"), RECV).len(), 1);
    assert_eq!(c.parse(&mk("l2_data"), RECV).len(), 1);
}

#[test]
fn coinbase_ignores_subscriptions_frame() {
    let raw = r#"{"channel":"subscriptions","sequence_num":0,
        "events":[{"subscriptions":{"level2":["BTC-USD"]}}]}"#;
    assert!(CoinbaseCollector::new().parse(raw, RECV).is_empty());
}

// ── Kraken ───────────────────────────────────────────────────────────────────

#[test]
fn kraken_ticker() {
    let raw = r#"{"channel":"ticker","type":"update","data":[{
        "symbol":"BTC/USD","bid":111999.1,"bid_qty":0.75,"ask":112000.4,"ask_qty":1.2,
        "last":112000.0,"volume":1234.5}]}"#;
    let b = one_bbo(KrakenCollector::new().parse(raw, RECV));
    assert_eq!(b.symbol, "btc/usd");
    assert_eq!(b.bid, d("111999.1"));
    assert_eq!(b.ask_qty, d("1.2"));
    // Kraken's ticker frame carries no timestamp
    assert_eq!(b.ts_ex, RECV);
}

#[test]
fn kraken_trade() {
    let raw = r#"{"channel":"trade","type":"update","data":[{
        "symbol":"ETH/USD","side":"sell","price":3011.25,"qty":0.5,
        "ord_type":"market","trade_id":42,"timestamp":"2026-08-11T09:33:19.750000Z"}]}"#;
    let t = trades(KrakenCollector::new().parse(raw, RECV));
    assert_eq!(t.len(), 1);
    assert_eq!(t[0].symbol, "eth/usd");
    assert_eq!(t[0].side, TradeSide::Sell);
    assert_eq!(t[0].price, d("3011.25"));
    assert_eq!(t[0].ts_ex, 1_786_440_799_750);
    assert_eq!(t[0].trade_id, "42");
}

#[test]
fn kraken_book_object_levels() {
    // Kraken sends levels as {"price":..,"qty":..} objects, not [p,q] arrays.
    let raw = r#"{"channel":"book","type":"snapshot","data":[{
        "symbol":"SOL/USD",
        "bids":[{"price":180.10,"qty":12.0},{"price":180.09,"qty":3.5}],
        "asks":[{"price":180.20,"qty":8.0}],
        "checksum":3037960894,"timestamp":"2026-08-11T09:33:20.000000Z"}]}"#;
    let b = one_book(KrakenCollector::new().parse(raw, RECV));
    assert_eq!(b.symbol, "sol/usd");
    assert!(b.is_snapshot);
    assert_eq!(b.bids, vec![[d("180.10"), d("12.0")], [d("180.09"), d("3.5")]]);
    assert_eq!(b.asks, vec![[d("180.20"), d("8.0")]]);
    assert_eq!(b.ts_ex, 1_786_440_800_000);
    assert_eq!(b.seq, Some(3_037_960_894));
}

#[test]
fn kraken_book_update_not_snapshot() {
    let raw = r#"{"channel":"book","type":"update","data":[{
        "symbol":"XRP/USD","bids":[{"price":2.50,"qty":0.0}],"asks":[],
        "checksum":1,"timestamp":"2026-08-11T09:33:21.000000Z"}]}"#;
    let b = one_book(KrakenCollector::new().parse(raw, RECV));
    assert!(!b.is_snapshot);
    assert_eq!(b.bids, vec![[d("2.50"), Decimal::ZERO]]);
}

#[test]
fn kraken_ignores_control_frames() {
    let c = KrakenCollector::new();
    assert!(c.parse(r#"{"method":"pong","req_id":1,"time_in":"x","time_out":"y"}"#, RECV).is_empty());
    assert!(c.parse(r#"{"channel":"heartbeat"}"#, RECV).is_empty());
    assert!(c.parse(r#"{"method":"subscribe","result":{"channel":"ticker"},"success":true}"#, RECV).is_empty());
    assert!(c.parse(r#"{"channel":"status","type":"update","data":[{"version":"2.0.11"}]}"#, RECV).is_empty());
}

// ── cross-venue invariants ───────────────────────────────────────────────────

#[test]
fn malformed_json_never_panics() {
    let specs: Vec<Box<dyn ExchangeSpec>> = vec![
        Box::new(BinanceCollector::new()),
        Box::new(OkxCollector::new()),
        Box::new(BybitCollector::new()),
        Box::new(CoinbaseCollector::new()),
        Box::new(KrakenCollector::new()),
    ];
    let junk = ["", "{", "null", "[]", "\"text\"", "{\"data\":null}", "{\"topic\":123}"];
    for spec in &specs {
        for j in junk {
            assert!(spec.parse(j, RECV).is_empty(), "{} choked on {j:?}", spec.name());
        }
    }
}

#[test]
fn every_spec_subscribes_and_declares_a_frame_cap() {
    let specs: Vec<Box<dyn ExchangeSpec>> = vec![
        Box::new(BinanceCollector::new()),
        Box::new(OkxCollector::new()),
        Box::new(BybitCollector::new()),
        Box::new(CoinbaseCollector::new()),
        Box::new(KrakenCollector::new()),
    ];
    for spec in &specs {
        assert!(spec.ws_url().starts_with("wss://"), "{} url must be TLS", spec.name());
        let msgs = spec.subscribe_msgs();
        assert!(!msgs.is_empty(), "{} sends no subscribe frames", spec.name());
        for m in &msgs {
            serde_json::from_str::<serde_json::Value>(m)
                .unwrap_or_else(|e| panic!("{} subscribe frame is not JSON: {e}", spec.name()));
        }
        assert!(spec.max_frame_size() >= 1 << 20, "{} frame cap too small", spec.name());
    }
}

#[test]
fn coinbase_asks_for_a_bigger_frame_cap() {
    // L2 snapshots for BTC-USD exceed the 2 MiB default.
    assert!(CoinbaseCollector::new().max_frame_size() > BinanceCollector::new().max_frame_size());
}
