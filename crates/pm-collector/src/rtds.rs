//! Polymarket RTDS —— Chainlink TWAP-30 采集器
//!
//! 协议：
//!   端点     : wss://ws-live-data.polymarket.com
//!   订阅     : {"action":"subscribe","subscriptions":[{"topic":"crypto_prices_twap_thirty","type":"update"}]}
//!   心跳     : 每 cfg.rtds_ping_s 秒发送文本 "PING"，服务端回复 "PONG"
//!   数据消息 : {"topic":"crypto_prices_twap_thirty","type":"update","timestamp":<pub_ms>,
//!               "payload":{"symbol":"btc/usd","full_accuracy_value":"<i128_str>",
//!                          "timestamp":<obs_ms>,"window_s":30}}
//!
//! 持久性：
//!   JSONL —— 所有 symbol，永远写入（RTDS 无法重放；约 2 MB/天）
//!   CH    —— 仅目标 symbol，批量写入

use crate::config::Config;
use crate::sink::Sink;
use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use pm_features::types::TwapRecord;
use rust_decimal::Decimal;
use std::str::FromStr;
use std::time::{Duration, Instant};
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use tokio_tungstenite::tungstenite::Message;

const TOPIC: &str = "crypto_prices_twap_thirty";
const WINDOW_S_DEFAULT: u16 = 30;

pub async fn run(cfg: Config) -> Result<()> {
    const HEALTHY: Duration = Duration::from_secs(60);
    let mut backoff = cfg.rtds_backoff_init;
    let client = crate::ch::make_client(&cfg);

    tracing::info!(
        "RTDS collector starting | url={} stale={}s backup={}",
        cfg.rtds_url, cfg.rtds_stale_s, cfg.backup_dir,
    );

    loop {
        let started = Instant::now();
        crate::ch::write_conn_event(&client, "rtds", "connecting", &cfg.rtds_url).await;

        match run_session(&cfg).await {
            Ok(reason) => {
                tracing::warn!("rtds session ended: {reason}");
                crate::ch::write_conn_event(&client, "rtds", "disconnected", &reason).await;
            }
            Err(e) => {
                tracing::error!("rtds session error: {e:#}");
                crate::ch::write_conn_event(&client, "rtds", "error", &e.to_string()).await;
            }
        }

        if started.elapsed() >= HEALTHY {
            backoff = cfg.rtds_backoff_init;
        }
        tracing::info!("rtds reconnecting in {backoff}s");
        tokio::time::sleep(Duration::from_secs(backoff)).await;
        backoff = (backoff * 2).min(cfg.rtds_backoff_max);
    }
}

// ── 会话生命周期 ──────────────────────────────────────────────────────────────

async fn run_session(cfg: &Config) -> Result<String> {
    let ws_cfg = WebSocketConfig::default()
        .max_frame_size(Some(1 << 20))   // 对 RTDS 文本帧来说 1 MiB 足够宽裕
        .max_message_size(Some(1 << 20));

    let (mut ws, _) =
        tokio_tungstenite::connect_async_with_config(&cfg.rtds_url, Some(ws_cfg), true).await?;
    tracing::info!("rtds connected to {}", cfg.rtds_url);

    // 订阅 —— 不加过滤条件，在客户端按 payload.symbol 分流
    let sub = serde_json::json!({
        "action": "subscribe",
        "subscriptions": [{"topic": TOPIC, "type": "update"}],
    });
    ws.send(Message::Text(sub.to_string().into())).await?;

    let sink           = crate::sink::spawn_writer(cfg);
    let jsonl          = Sink::new(cfg);   // 仅用于 JSONL 镜像，不写 ClickHouse
    let mut ping_tick  = tokio::time::interval(Duration::from_secs(cfg.rtds_ping_s.max(1)));
    let mut flush_tick = tokio::time::interval(Duration::from_secs(1));
    ping_tick.tick().await;
    flush_tick.tick().await;

    let stale    = Duration::from_secs(cfg.rtds_stale_s.max(1));
    let mut last_msg = Instant::now();
    let mut msgs: u64 = 0;
    let mut secs: u64 = 0;

    let reason = loop {
        tokio::select! {
            frame = ws.next() => {
                let Some(frame) = frame else { break "stream closed".into() };
                match frame? {
                    Message::Text(t) => {
                        last_msg = Instant::now();
                        if t.as_str() == "PONG" { continue; }
                        // 服务端偶尔发空帧或纯空白帧，忽略即可
                        if t.as_str().trim().is_empty() { continue; }
                        msgs += 1;
                        if let Some(rec) = parse_frame(t.as_str(), now_ms()) {
                            // JSONL：所有 symbol，无条件写入
                            jsonl.mirror_jsonl(&rec, "twap30");
                            // CH：仅目标 symbol
                            if cfg.is_target(&rec.symbol) {
                                sink.push_twap(&rec);
                            }
                        }
                    }
                    Message::Binary(_)
                    | Message::Ping(_)
                    | Message::Pong(_) => { last_msg = Instant::now(); }
                    Message::Close(c) => break format!("close frame: {c:?}"),
                    Message::Frame(_) => {}
                }
            }
            _ = ping_tick.tick() => {
                if let Err(e) = ws.send(Message::Text("PING".into())).await {
                    break format!("ping failed: {e}");
                }
            }
            _ = flush_tick.tick() => {
                if last_msg.elapsed() > stale {
                    break format!("stale: no frame for {}s", last_msg.elapsed().as_secs());
                }
                secs += 1;
                if secs % 60 == 0 {
                    tracing::info!("rtds: {msgs} msgs, {} dropped", sink.dropped_count());
                }
            }
        }
    };

    sink.request_flush();
    Ok(reason)
}

// ── 帧解析器 ──────────────────────────────────────────────────────────────────

fn parse_frame(raw: &str, recv_ms: i64) -> Option<TwapRecord> {
    let msg: serde_json::Value = serde_json::from_str(raw)
        .map_err(|e| tracing::warn!("rtds json parse error: {e} | raw={raw:.200}"))
        .ok()?;

    // 只处理我们订阅的 topic 上的数据更新
    if msg.get("topic")?.as_str()? != TOPIC { return None; }
    if msg.get("type")?.as_str()? != "update" { return None; }

    let payload = msg.get("payload")?.as_object()?;

    let symbol = payload.get("symbol")?.as_str()?.to_lowercase();
    if symbol.is_empty() { return None; }

    let value_e18 = payload.get("full_accuracy_value")?.as_str()?.trim().to_owned();
    if value_e18.is_empty() { return None; }

    // 校验：必须能解析为正整数
    let parsed_i128: i128 = value_e18.parse()
        .map_err(|e| tracing::warn!("rtds value_e18 无效 ({e}): {value_e18:?}"))
        .ok()?;
    if parsed_i128 <= 0 {
        tracing::warn!("rtds value_e18 非正数: {value_e18}");
        return None;
    }

    let obs_ms: i64 = payload.get("timestamp")
        .and_then(|v| v.as_i64())
        .unwrap_or(recv_ms);

    let pub_ms: i64 = msg.get("timestamp")
        .and_then(|v| v.as_i64())
        .unwrap_or(recv_ms);

    let window_s: u16 = payload.get("window_s")
        .and_then(|v| v.as_u64())
        .map(|v| v as u16)
        .unwrap_or(WINDOW_S_DEFAULT);

    // price = value_e18 / 1e18 —— 用 Decimal 获得精确表示
    let price = Decimal::from_str(&value_e18).ok()
        .and_then(|v| Decimal::from_str("1000000000000000000").ok().map(|e18| v / e18))
        .unwrap_or_default();

    Some(TwapRecord { symbol, window_s, obs_ms, pub_ms, recv_ms, value_e18, price })
}

#[inline]
fn now_ms() -> i64 { chrono::Utc::now().timestamp_millis() }

#[cfg(test)]
mod tests {
    use super::*;

    const RECV: i64 = 1_754_900_000_000;

    fn frame(sym: &str, val: &str) -> String {
        format!(r#"{{"topic":"crypto_prices_twap_thirty","type":"update","timestamp":1754899999500,
            "payload":{{"symbol":"{sym}","full_accuracy_value":"{val}",
                        "timestamp":1754899999000,"window_s":30}}}}"#)
    }

    #[test]
    fn parses_a_twap_update() {
        let r = parse_frame(&frame("BTC/USD", "112345670000000000000000"), RECV).unwrap();
        assert_eq!(r.symbol, "btc/usd", "symbol 必须转为小写");
        assert_eq!(r.window_s, 30);
        assert_eq!(r.obs_ms, 1_754_899_999_000, "obs_ms 取自 payload.timestamp");
        assert_eq!(r.pub_ms, 1_754_899_999_500, "pub_ms 取自外层信封的 timestamp");
        assert_eq!(r.recv_ms, RECV);
        assert_eq!(r.value_e18, "112345670000000000000000");
        // 112345670000000000000000 / 1e18 = 112345.67
        assert_eq!(r.price, Decimal::from_str("112345.67").unwrap());
    }

    #[test]
    fn price_keeps_sub_cent_precision() {
        // DOGE 价格在 $0.22 附近；有意义的数字在小数点右边很远的位置，
        // 必须在 /1e18 除法后仍然保留。
        let r = parse_frame(&frame("DOGE/USD", "220145678900000000"), RECV).unwrap();
        assert_eq!(r.price, Decimal::from_str("0.2201456789").unwrap());
    }

    #[test]
    fn rejects_bad_payloads() {
        // 非数字、零、负数、缺失值均应丢弃该记录
        assert!(parse_frame(&frame("BTC/USD", "not-a-number"), RECV).is_none());
        assert!(parse_frame(&frame("BTC/USD", "0"), RECV).is_none());
        assert!(parse_frame(&frame("BTC/USD", "-1"), RECV).is_none());
        assert!(parse_frame(&frame("BTC/USD", ""), RECV).is_none());
    }

    #[test]
    fn ignores_other_topics_and_types() {
        let other_topic = r#"{"topic":"crypto_prices","type":"update","timestamp":1,
            "payload":{"symbol":"btc/usd","full_accuracy_value":"1","timestamp":1}}"#;
        assert!(parse_frame(other_topic, RECV).is_none());

        let other_type = r#"{"topic":"crypto_prices_twap_thirty","type":"subscribed","timestamp":1,
            "payload":{"symbol":"btc/usd","full_accuracy_value":"1","timestamp":1}}"#;
        assert!(parse_frame(other_type, RECV).is_none());
    }

    #[test]
    fn missing_timestamps_fall_back_to_recv() {
        let raw = r#"{"topic":"crypto_prices_twap_thirty","type":"update",
            "payload":{"symbol":"eth/usd","full_accuracy_value":"3000000000000000000000"}}"#;
        let r = parse_frame(raw, RECV).unwrap();
        assert_eq!(r.obs_ms, RECV);
        assert_eq!(r.pub_ms, RECV);
        assert_eq!(r.window_s, WINDOW_S_DEFAULT, "window_s 默认为 30");
    }

    #[test]
    fn malformed_input_never_panics() {
        for j in ["", "{", "null", "[]", "\"PONG\"", "{\"topic\":123}", "{\"payload\":[]}"] {
            assert!(parse_frame(j, RECV).is_none(), "在 {j:?} 上出错");
        }
    }
}
