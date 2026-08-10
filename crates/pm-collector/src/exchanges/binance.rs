use super::{util, ExchangeSpec, ParsedRecord};
use pm_features::types::{BboRecord, BookRecord, Exchange, TradeRecord};
use serde_json::Value;

const SYM: &[(&str, &str)] = &[
    ("btcusdt",  "btc/usd"),
    ("ethusdt",  "eth/usd"),
    ("solusdt",  "sol/usd"),
    ("xrpusdt",  "xrp/usd"),
    ("dogeusdt", "doge/usd"),
];

pub struct BinanceCollector {
    ws_url: String,
}

impl BinanceCollector {
    pub fn new() -> Self {
        Self { ws_url: "wss://stream.binance.com:9443/stream".into() }
    }
}

impl ExchangeSpec for BinanceCollector {
    fn name(&self) -> &'static str { "binance" }
    fn ws_url(&self) -> &str { &self.ws_url }

    fn subscribe_msgs(&self) -> Vec<String> {
        let streams: Vec<&str> = vec![
            "btcusdt@bookTicker","ethusdt@bookTicker","solusdt@bookTicker",
            "xrpusdt@bookTicker","dogeusdt@bookTicker",
            "btcusdt@aggTrade","ethusdt@aggTrade","solusdt@aggTrade",
            "xrpusdt@aggTrade","dogeusdt@aggTrade",
            "btcusdt@depth@100ms","ethusdt@depth@100ms","solusdt@depth@100ms",
            "xrpusdt@depth@100ms","dogeusdt@depth@100ms",
        ];
        vec![serde_json::json!({"method":"SUBSCRIBE","params":streams,"id":1}).to_string()]
    }

    fn ping_msg(&self) -> Option<String> { None } // 仅使用协议层 ping

    fn parse(&self, raw: &str, recv_ms: i64) -> Vec<ParsedRecord> {
        let msg: Value = match serde_json::from_str(raw) { Ok(v) => v, Err(_) => return vec![] };

        let stream = match msg["stream"].as_str() { Some(s) => s, None => return vec![] };
        let data   = &msg["data"];
        let sym_raw = stream.split('@').next().unwrap_or("");
        let symbol  = match util::sym_lookup(SYM, sym_raw) { Some(s) => s.to_owned(), None => return vec![] };

        if stream.contains("bookTicker") {
            // bookTicker 不带事件时间 —— 用 recv_ms 作为 ts_ex
            let bid = match util::dec(&data["b"]) { Some(v) => v, None => return vec![] };
            let ask = match util::dec(&data["a"]) { Some(v) => v, None => return vec![] };
            return vec![ParsedRecord::Bbo(BboRecord {
                exchange: Exchange::Binance,
                symbol,
                ts_ex:   recv_ms,
                recv_ms,
                bid,
                bid_qty: util::dec_or_zero(&data["B"]),
                ask,
                ask_qty: util::dec_or_zero(&data["A"]),
            })];
        }

        if stream.contains("aggTrade") {
            // m=true → maker 是买方 → taker 是卖方
            let is_maker_buyer = data["m"].as_bool().unwrap_or(false);
            let side = if is_maker_buyer { pm_features::types::TradeSide::Sell }
                       else              { pm_features::types::TradeSide::Buy  };
            return vec![ParsedRecord::Trade(TradeRecord {
                exchange: Exchange::Binance,
                symbol,
                ts_ex:    util::ts(Some(&data["T"]), recv_ms),
                recv_ms,
                price:    match util::dec(&data["p"]) { Some(v) => v, None => return vec![] },
                qty:      util::dec_or_zero(&data["q"]),
                side,
                trade_id: util::s(Some(&data["a"])),
            })];
        }

        if stream.contains("depth") {
            return vec![ParsedRecord::Book(BookRecord {
                exchange:    Exchange::Binance,
                symbol,
                ts_ex:       util::ts(Some(&data["T"]), recv_ms),
                recv_ms,
                bids:        util::ladder(Some(&data["b"])),
                asks:        util::ladder(Some(&data["a"])),
                is_snapshot: false,
                seq:         data["u"].as_u64(),
            })];
        }

        vec![]
    }
}
