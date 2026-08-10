use super::{util, ExchangeSpec, ParsedRecord};
use pm_features::types::{BboRecord, BookRecord, Exchange, TradeRecord};
use serde_json::Value;

const MAX_ARGS: usize = 10; // Bybit 拒绝单帧包含超过 10 个参数的订阅请求

const SYM: &[(&str, &str)] = &[
    ("BTCUSDT",  "btc/usd"),
    ("ETHUSDT",  "eth/usd"),
    ("SOLUSDT",  "sol/usd"),
    ("XRPUSDT",  "xrp/usd"),
    ("DOGEUSDT", "doge/usd"),
];

pub struct BybitCollector {
    ws_url: String,
}

impl BybitCollector {
    pub fn new() -> Self {
        Self { ws_url: "wss://stream.bybit.com/v5/public/spot".into() }
    }
}

impl ExchangeSpec for BybitCollector {
    fn name(&self) -> &'static str { "bybit" }
    fn ws_url(&self) -> &str { &self.ws_url }

    fn subscribe_msgs(&self) -> Vec<String> {
        let topics: Vec<&str> = vec![
            "orderbook.1.BTCUSDT","orderbook.1.ETHUSDT","orderbook.1.SOLUSDT",
            "orderbook.1.XRPUSDT","orderbook.1.DOGEUSDT",
            "publicTrade.BTCUSDT","publicTrade.ETHUSDT","publicTrade.SOLUSDT",
            "publicTrade.XRPUSDT","publicTrade.DOGEUSDT",
            "orderbook.50.BTCUSDT","orderbook.50.ETHUSDT","orderbook.50.SOLUSDT",
            "orderbook.50.XRPUSDT","orderbook.50.DOGEUSDT",
        ];
        topics.chunks(MAX_ARGS)
            .map(|chunk| serde_json::json!({"op":"subscribe","args":chunk}).to_string())
            .collect()
    }

    fn ping_msg(&self) -> Option<String> {
        Some(serde_json::json!({"op":"ping"}).to_string())
    }

    fn parse(&self, raw: &str, recv_ms: i64) -> Vec<ParsedRecord> {
        let msg: Value = match serde_json::from_str(raw) { Ok(v) => v, Err(_) => return vec![] };

        // 订阅 ack / pong 帧带有 "op" 但没有 "topic"
        let topic = match msg["topic"].as_str() { Some(t) => t, None => return vec![] };
        let data  = &msg["data"];
        let ts    = util::ts(Some(&msg["ts"]), recv_ms);

        // orderbook.1.<SYM> —— 当作 BBO 处理
        if let Some(rest) = topic.strip_prefix("orderbook.1.") {
            let symbol = match util::sym_lookup(SYM, rest) { Some(s) => s.to_owned(), None => return vec![] };
            let (bid, bid_qty) = util::top_level(Some(&data["b"])).unwrap_or_default();
            let (ask, ask_qty) = util::top_level(Some(&data["a"])).unwrap_or_default();
            // Bybit 的 depth-1 增量可能某一边为空；跳过这类帧，
            // 价格为零的 BBO 行会污染 spread/mid 特征。
            if bid.is_zero() || ask.is_zero() { return vec![]; }
            return vec![ParsedRecord::Bbo(BboRecord {
                exchange: Exchange::Bybit, symbol, ts_ex: ts, recv_ms,
                bid, bid_qty, ask, ask_qty,
            })];
        }

        if let Some(rest) = topic.strip_prefix("publicTrade.") {
            let symbol = match util::sym_lookup(SYM, rest) { Some(s) => s.to_owned(), None => return vec![] };
            let Some(rows) = data.as_array() else { return vec![] };
            return rows.iter().filter_map(|t| {
                let side = util::side(t["S"].as_str().unwrap_or(""))?;
                Some(ParsedRecord::Trade(TradeRecord {
                    exchange: Exchange::Bybit,
                    symbol: symbol.clone(),
                    ts_ex:    util::ts(Some(&t["T"]), ts),
                    recv_ms,
                    price:    util::dec(&t["p"])?,
                    qty:      util::dec_or_zero(&t["v"]),
                    side,
                    trade_id: util::s(Some(&t["i"])),
                }))
            }).collect();
        }

        if let Some(rest) = topic.strip_prefix("orderbook.50.") {
            let symbol = match util::sym_lookup(SYM, rest) { Some(s) => s.to_owned(), None => return vec![] };
            // u == 1 表示服务重启：Bybit 会重置盘口，因此即使 type 标为
            // "delta"，这一帧也必须按快照来应用。
            let is_snapshot = msg["type"].as_str() == Some("snapshot")
                           || data["u"].as_u64() == Some(1);
            return vec![ParsedRecord::Book(BookRecord {
                exchange:    Exchange::Bybit,
                symbol,
                ts_ex:       ts,
                recv_ms,
                bids:        util::ladder(Some(&data["b"])),
                asks:        util::ladder(Some(&data["a"])),
                is_snapshot,
                seq:         data["seq"].as_u64().or_else(|| data["u"].as_u64()),
            })];
        }

        vec![]
    }
}
