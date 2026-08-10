use super::{util, ExchangeSpec, ParsedRecord};
use pm_features::types::{BboRecord, BookRecord, Exchange, TradeRecord};
use rust_decimal::Decimal;
use serde_json::Value;

const SYM: &[(&str, &str)] = &[
    ("BTC/USD",  "btc/usd"),
    ("ETH/USD",  "eth/usd"),
    ("SOL/USD",  "sol/usd"),
    ("XRP/USD",  "xrp/usd"),
    ("DOGE/USD", "doge/usd"),
];

pub struct KrakenCollector {
    ws_url: String,
}

impl KrakenCollector {
    pub fn new() -> Self {
        Self { ws_url: "wss://ws.kraken.com/v2".into() }
    }
}

impl ExchangeSpec for KrakenCollector {
    fn name(&self) -> &'static str { "kraken" }
    fn ws_url(&self) -> &str { &self.ws_url }

    fn subscribe_msgs(&self) -> Vec<String> {
        let symbols = ["BTC/USD","ETH/USD","SOL/USD","XRP/USD","DOGE/USD"];
        vec![
            serde_json::json!({"method":"subscribe","params":{"channel":"ticker","symbol":symbols}}).to_string(),
            serde_json::json!({"method":"subscribe","params":{"channel":"trade","symbol":symbols}}).to_string(),
            serde_json::json!({"method":"subscribe","params":{"channel":"book","symbol":symbols,"depth":10,"snapshot":true}}).to_string(),
        ]
    }

    fn ping_msg(&self) -> Option<String> {
        Some(serde_json::json!({"method":"ping"}).to_string())
    }

    fn parse(&self, raw: &str, recv_ms: i64) -> Vec<ParsedRecord> {
        let msg: Value = match serde_json::from_str(raw) { Ok(v) => v, Err(_) => return vec![] };

        // method 响应帧（订阅 ack、pong、heartbeat）
        if msg["method"].is_string() || msg["channel"].as_str() == Some("heartbeat") {
            return vec![];
        }

        // Kraken v2 把数据源名称放在 "channel"，把 snapshot/update 放在
        // "type" —— 这是两个独立字段，不是同一个。
        let channel = match msg["channel"].as_str() { Some(c) => c, None => return vec![] };
        let Some(data_arr) = msg["data"].as_array() else { return vec![] };

        let mut out = Vec::new();

        match channel {
            "ticker" => {
                for d in data_arr {
                    let Some(symbol) = util::sym_lookup(SYM, d["symbol"].as_str().unwrap_or("")) else { continue };
                    // Kraken 的 ticker 不带时间戳，使用 recv_ms
                    let Some(bid) = util::dec(&d["bid"]) else { continue };
                    let Some(ask) = util::dec(&d["ask"]) else { continue };
                    out.push(ParsedRecord::Bbo(BboRecord {
                        exchange: Exchange::Kraken,
                        symbol:   symbol.to_owned(),
                        ts_ex:    recv_ms,
                        recv_ms,
                        bid,
                        bid_qty:  util::dec_or_zero(&d["bid_qty"]),
                        ask,
                        ask_qty:  util::dec_or_zero(&d["ask_qty"]),
                    }));
                }
            }
            "trade" => {
                for t in data_arr {
                    let Some(symbol) = util::sym_lookup(SYM, t["symbol"].as_str().unwrap_or("")) else { continue };
                    let Some(side)   = util::side(t["side"].as_str().unwrap_or("")) else { continue };
                    let Some(price)  = util::dec(&t["price"]) else { continue };
                    out.push(ParsedRecord::Trade(TradeRecord {
                        exchange: Exchange::Kraken,
                        symbol:   symbol.to_owned(),
                        ts_ex:    util::iso_ms(Some(&t["timestamp"]), recv_ms),
                        recv_ms,
                        price,
                        qty:      util::dec_or_zero(&t["qty"]),
                        side,
                        trade_id: util::s(Some(&t["trade_id"])),
                    }));
                }
            }
            "book" => {
                for d in data_arr {
                    let Some(symbol) = util::sym_lookup(SYM, d["symbol"].as_str().unwrap_or("")) else { continue };
                    // Kraken 的盘口档位是 {"price": f, "qty": f} 这样的对象
                    let parse_levels = |lvls: Option<&Value>| -> Vec<[Decimal; 2]> {
                        lvls.and_then(|v| v.as_array())
                            .map(|arr| arr.iter().filter_map(|l| {
                                Some([util::dec(&l["price"])?, util::dec_or_zero(&l["qty"])])
                            }).collect())
                            .unwrap_or_default()
                    };
                    let is_snapshot = msg["type"].as_str() == Some("snapshot");
                    out.push(ParsedRecord::Book(BookRecord {
                        exchange:    Exchange::Kraken,
                        symbol:      symbol.to_owned(),
                        ts_ex:       util::iso_ms(Some(&d["timestamp"]), recv_ms),
                        recv_ms,
                        bids:        parse_levels(Some(&d["bids"])),
                        asks:        parse_levels(Some(&d["asks"])),
                        is_snapshot,
                        seq:         d["checksum"].as_u64(), // 复用 seq 字段存放完整性校验和
                    }));
                }
            }
            _ => {}
        }

        out
    }
}
