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

/// REST 底图的深度档数。1000 是 binance 允许的最大值，也是唯一能让
/// 深度分布类特征（加权深度斜率等）有意义的档数——20 档只覆盖盘口
/// 附近几个基点，看不到真正的流动性结构。
const SNAPSHOT_LIMIT: u32 = 1000;

pub struct BinanceCollector {
    ws_url: String,
    http:   reqwest::Client,
}

impl BinanceCollector {
    pub fn new() -> Self {
        Self {
            ws_url: "wss://stream.binance.com:9443/stream".into(),
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("构建 HTTP 客户端失败"),
        }
    }

    /// 拉取单个 symbol 的 L2 底图。
    async fn fetch_depth(&self, inst: &str, symbol: &str) -> Option<BookRecord> {
        let url = format!(
            "https://api.binance.com/api/v3/depth?symbol={}&limit={SNAPSHOT_LIMIT}",
            inst.to_uppercase()
        );
        let resp = match self.http.get(&url).send().await {
            Ok(r) => r,
            Err(e) => { tracing::warn!("binance depth {inst} 请求失败: {e}"); return None; }
        };
        if !resp.status().is_success() {
            tracing::warn!("binance depth {inst} 返回 {}", resp.status());
            return None;
        }
        let v: Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => { tracing::warn!("binance depth {inst} JSON 解析失败: {e}"); return None; }
        };

        let last_update_id = v["lastUpdateId"].as_u64()?;
        let recv_ms = chrono::Utc::now().timestamp_millis();

        Some(BookRecord {
            exchange:    Exchange::Binance,
            symbol:      symbol.to_owned(),
            // REST 快照不带交易所时间戳，用接收时刻——这也是快照真正
            // 对我们生效的时刻。
            ts_ex:       recv_ms,
            recv_ms,
            bids:        util::ladder(Some(&v["bids"])),
            asks:        util::ladder(Some(&v["asks"])),
            is_snapshot: true,
            // 增量的连续性条件是 U <= lastUpdateId+1 <= u，所以两个
            // 序列字段都填 lastUpdateId，离线重放时按这个起点对齐。
            seq:         Some(last_update_id),
            first_seq:   Some(last_update_id),
        })
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

    fn fetch_snapshots<'a>(&'a self)
        -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<BookRecord>> + Send + 'a>>
    {
        Box::pin(async move {
            let mut out = Vec::with_capacity(SYM.len());
            for (inst, symbol) in SYM {
                if let Some(rec) = self.fetch_depth(inst, symbol).await {
                    out.push(rec);
                }
                // binance REST 有权重限制（depth limit=1000 计 50 权重，
                // 每分钟上限 6000）。五个 symbol 共 250 权重，间隔 200ms
                // 已远低于限制，但重连风暴下仍需避免瞬间打满。
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            }
            out
        })
    }

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
                first_seq:   data["U"].as_u64(),
            })];
        }

        vec![]
    }
}
