use super::{util, ExchangeSpec, ParsedRecord};
use pm_features::types::{BboRecord, BookRecord, Exchange, TradeRecord};
use serde_json::Value;

const SYM: &[(&str, &str)] = &[
    ("BTC-USDT",  "btc/usd"),
    ("ETH-USDT",  "eth/usd"),
    ("SOL-USDT",  "sol/usd"),
    ("XRP-USDT",  "xrp/usd"),
    ("DOGE-USDT", "doge/usd"),
];

pub struct OkxCollector {
    ws_url: String,
}

impl OkxCollector {
    pub fn new() -> Self {
        Self { ws_url: "wss://ws.okx.com:8443/ws/v5/public".into() }
    }
}

impl ExchangeSpec for OkxCollector {
    fn name(&self) -> &'static str { "okx" }
    fn ws_url(&self) -> &str { &self.ws_url }

    fn subscribe_msgs(&self) -> Vec<String> {
        let args: Vec<serde_json::Value> = ["BTC-USDT","ETH-USDT","SOL-USDT","XRP-USDT","DOGE-USDT"]
            .iter()
            .flat_map(|inst| [
                serde_json::json!({"channel":"bbo-tbt","instId":inst}),
                serde_json::json!({"channel":"trades","instId":inst}),
                serde_json::json!({"channel":"books5","instId":inst}),
            ])
            .collect();
        vec![serde_json::json!({"op":"subscribe","args":args}).to_string()]
    }

    fn ping_msg(&self) -> Option<String> { Some("ping".into()) }

    fn parse(&self, raw: &str, recv_ms: i64) -> Vec<ParsedRecord> {
        if raw == "pong" { return vec![]; }
        let msg: Value = match serde_json::from_str(raw) { Ok(v) => v, Err(_) => return vec![] };

        // 订阅确认 / 错误帧
        if msg["event"].is_string() { return vec![]; }

        let ch     = match msg["arg"]["channel"].as_str() { Some(v) => v, None => return vec![] };
        let inst   = msg["arg"]["instId"].as_str().unwrap_or("");
        let symbol = match util::sym_lookup(SYM, inst) { Some(s) => s.to_owned(), None => return vec![] };

        let data_list = match msg["data"].as_array() { Some(a) => a, None => return vec![] };
        if data_list.is_empty() { return vec![]; }

        match ch {
            "bbo-tbt" => {
                let d = &data_list[0];
                let ts = util::ts(Some(&d["ts"]), recv_ms);
                let (bid, bid_qty) = util::top_level(Some(&d["bids"])).unwrap_or_default();
                let (ask, ask_qty) = util::top_level(Some(&d["asks"])).unwrap_or_default();
                vec![ParsedRecord::Bbo(BboRecord {
                    exchange: Exchange::Okx, symbol, ts_ex: ts, recv_ms,
                    bid, bid_qty, ask, ask_qty,
                })]
            }
            "trades" => {
                data_list.iter().filter_map(|t| {
                    let side = util::side(t["side"].as_str().unwrap_or(""))?;
                    Some(ParsedRecord::Trade(TradeRecord {
                        exchange: Exchange::Okx,
                        symbol: symbol.clone(),
                        ts_ex:    util::ts(Some(&t["ts"]), recv_ms),
                        recv_ms,
                        price:    util::dec(&t["px"])?,
                        qty:      util::dec_or_zero(&t["sz"]),
                        side,
                        trade_id: util::s(Some(&t["tradeId"])),
                    }))
                }).collect()
            }
            "books5" => {
                let d = &data_list[0];
                vec![ParsedRecord::Book(BookRecord {
                    exchange:    Exchange::Okx,
                    symbol,
                    ts_ex:       util::ts(Some(&d["ts"]), recv_ms),
                    recv_ms,
                    bids:        util::ladder(Some(&d["bids"])),
                    asks:        util::ladder(Some(&d["asks"])),
                    is_snapshot: true, // books5 每次都发送完整快照
                    seq:         d["seqId"].as_u64(),
                })]
            }
            _ => vec![],
        }
    }
}
