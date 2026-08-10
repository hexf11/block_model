use super::{util, ExchangeSpec, ParsedRecord};
use pm_features::types::{BboRecord, BookRecord, Exchange, TradeRecord};
use rust_decimal::Decimal;
use serde_json::Value;

const SYM: &[(&str, &str)] = &[
    ("BTC-USD",  "btc/usd"),
    ("ETH-USD",  "eth/usd"),
    ("SOL-USD",  "sol/usd"),
    ("XRP-USD",  "xrp/usd"),
    ("DOGE-USD", "doge/usd"),
];

pub struct CoinbaseCollector {
    ws_url: String,
}

impl CoinbaseCollector {
    pub fn new() -> Self {
        Self { ws_url: "wss://advanced-trade-ws.coinbase.com".into() }
    }
}

impl ExchangeSpec for CoinbaseCollector {
    fn name(&self) -> &'static str { "coinbase" }
    fn ws_url(&self) -> &str { &self.ws_url }

    fn subscribe_msgs(&self) -> Vec<String> {
        let product_ids = ["BTC-USD","ETH-USD","SOL-USD","XRP-USD","DOGE-USD"];
        ["ticker","market_trades","level2"].iter().map(|channel| {
            serde_json::json!({
                "type": "subscribe",
                "product_ids": product_ids,
                "channel": channel,
            }).to_string()
        }).collect()
    }

    // level2 快照会超过默认的 1 MiB 帧上限
    fn max_frame_size(&self) -> usize { 1 << 23 } // 8 MiB

    fn ping_msg(&self) -> Option<String> { None }

    fn parse(&self, raw: &str, recv_ms: i64) -> Vec<ParsedRecord> {
        let msg: Value = match serde_json::from_str(raw) { Ok(v) => v, Err(_) => return vec![] };
        let channel = msg["channel"].as_str().unwrap_or("");
        let Some(events) = msg["events"].as_array() else { return vec![] };

        let mut out = Vec::new();

        match channel {
            "ticker" => {
                for ev in events {
                    let Some(tickers) = ev["tickers"].as_array() else { continue };
                    for t in tickers {
                        let Some(symbol) = util::sym_lookup(SYM, t["product_id"].as_str().unwrap_or("")) else { continue };
                        let Some(bid) = util::dec(&t["best_bid"]) else { continue };
                        let Some(ask) = util::dec(&t["best_ask"]) else { continue };
                        out.push(ParsedRecord::Bbo(BboRecord {
                            exchange: Exchange::Coinbase,
                            symbol:   symbol.to_owned(),
                            ts_ex:    util::iso_ms(Some(&msg["timestamp"]), recv_ms),
                            recv_ms,
                            bid,
                            bid_qty:  util::dec_or_zero(&t["best_bid_quantity"]),
                            ask,
                            ask_qty:  util::dec_or_zero(&t["best_ask_quantity"]),
                        }));
                    }
                }
            }
            "market_trades" => {
                for ev in events {
                    let Some(trades) = ev["trades"].as_array() else { continue };
                    for t in trades {
                        let Some(symbol) = util::sym_lookup(SYM, t["product_id"].as_str().unwrap_or("")) else { continue };
                        let Some(side)   = util::side(t["side"].as_str().unwrap_or("")) else { continue };
                        let Some(price)  = util::dec(&t["price"]) else { continue };
                        out.push(ParsedRecord::Trade(TradeRecord {
                            exchange: Exchange::Coinbase,
                            symbol:   symbol.to_owned(),
                            ts_ex:    util::iso_ms(Some(&t["time"]), recv_ms),
                            recv_ms,
                            price,
                            qty:      util::dec_or_zero(&t["size"]),
                            side,
                            trade_id: util::s(Some(&t["trade_id"])),
                        }));
                    }
                }
            }
            // 订阅时用的是 "level2"，但收到的帧标记为 "l2_data"。
            "level2" | "l2_data" => {
                for ev in events {
                    let Some(symbol) = util::sym_lookup(SYM, ev["product_id"].as_str().unwrap_or("")) else { continue };
                    let is_snapshot = ev["type"].as_str() == Some("snapshot");
                    let mut bids = Vec::new();
                    let mut asks = Vec::new();
                    let Some(updates) = ev["updates"].as_array() else { continue };
                    for u in updates {
                        let Some(price) = util::dec(&u["price_level"]) else { continue };
                        let qty: Decimal = util::dec_or_zero(&u["new_quantity"]);
                        match u["side"].as_str().unwrap_or("") {
                            "bid" => bids.push([price, qty]),
                            // Coinbase 把卖方（ask）写成 "offer"
                            "offer" | "ask" => asks.push([price, qty]),
                            _ => {}
                        }
                    }
                    out.push(ParsedRecord::Book(BookRecord {
                        exchange: Exchange::Coinbase,
                        symbol:   symbol.to_owned(),
                        ts_ex:    util::iso_ms(Some(&msg["timestamp"]), recv_ms),
                        recv_ms,
                        bids,
                        asks,
                        is_snapshot,
                        seq:      msg["sequence_num"].as_u64(),
                    }));
                }
            }
            _ => {}
        }

        out
    }
}
