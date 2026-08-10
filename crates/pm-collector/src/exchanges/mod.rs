pub mod binance;
pub mod bybit;
pub mod coinbase;
pub mod kraken;
pub mod okx;
pub(crate) mod util;

use crate::config::Config;
use crate::sink::Sink;
use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use pm_features::types::{BboRecord, BookRecord, TradeRecord};
use std::time::{Duration, Instant};
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use tokio_tungstenite::tungstenite::Message;

/// 解析一个 WebSocket 帧后产出的记录。
pub enum ParsedRecord {
    Bbo(BboRecord),
    Trade(TradeRecord),
    Book(BookRecord),
}

/// 每家交易所实现这个 trait。本模块中的通用会话循环负责
/// 连接 / 订阅 / 重连 / 失活检测 / 批量刷新。
pub trait ExchangeSpec: Send + Sync + 'static {
    /// 例如 "binance"
    fn name(&self) -> &'static str;

    /// WebSocket 端点
    fn ws_url(&self) -> &str;

    /// 连接后要发送的订阅帧（受参数条数限制，可能有多条）
    fn subscribe_msgs(&self) -> Vec<String>;

    /// 将一个原始文本帧解析为零条或多条记录
    fn parse(&self, raw: &str, recv_ms: i64) -> Vec<ParsedRecord>;

    /// 可选的应用层 ping 帧；None 表示依赖协议层的 Ping
    fn ping_msg(&self) -> Option<String> { None }

    /// 入向 WS 帧/消息的最大尺寸。Coinbase 的 L2 快照需要 8 MiB。
    fn max_frame_size(&self) -> usize { 1 << 21 }
}

// ── 公共入口 ────────────────────────────────────────────────────────────────

/// 为每家交易所派生一个常驻任务。每个任务独占自己的 socket 和自己的
/// Sink，因此某个缓慢或已挂掉的交易所不会拖住其他交易所。
pub async fn run_all(cfg: Config) -> Result<()> {
    let specs: Vec<Box<dyn ExchangeSpec>> = vec![
        Box::new(binance::BinanceCollector::new()),
        Box::new(okx::OkxCollector::new()),
        Box::new(bybit::BybitCollector::new()),
        Box::new(coinbase::CoinbaseCollector::new()),
        Box::new(kraken::KrakenCollector::new()),
    ];

    let mut tasks = Vec::with_capacity(specs.len());
    for spec in specs {
        let cfg = cfg.clone();
        tasks.push(tokio::spawn(run_forever(spec, cfg)));
    }
    for t in tasks {
        if let Err(e) = t.await {
            tracing::error!("交易所任务 panic: {e}");
        }
    }
    Ok(())
}

// ── 重连循环 ────────────────────────────────────────────────────────────────

/// 采用指数退避无限重连。一旦某次会话健康运行满 60 秒，退避时间就重置，
/// 这样单次抖动不会惩罚之后的连接。
async fn run_forever(spec: Box<dyn ExchangeSpec>, cfg: Config) {
    const HEALTHY: Duration = Duration::from_secs(60);
    let mut backoff = cfg.ex_backoff_init;
    let client = crate::ch::make_client(&cfg);

    loop {
        let started = Instant::now();
        crate::ch::write_conn_event(&client, spec.name(), "connecting", "").await;

        let outcome = run_session(spec.as_ref(), &cfg).await;
        match &outcome {
            Ok(reason) => {
                tracing::warn!("{} session ended: {reason}", spec.name());
                crate::ch::write_conn_event(&client, spec.name(), "disconnected", reason).await;
            }
            Err(e) => {
                tracing::error!("{} session error: {e:#}", spec.name());
                crate::ch::write_conn_event(&client, spec.name(), "error", &e.to_string()).await;
            }
        }

        if started.elapsed() >= HEALTHY {
            backoff = cfg.ex_backoff_init;
        }
        tracing::info!("{} reconnecting in {backoff}s", spec.name());
        tokio::time::sleep(Duration::from_secs(backoff)).await;
        backoff = (backoff * 2).min(cfg.ex_backoff_max);
    }
}

// ── 会话生命周期 ─────────────────────────────────────────────────────────────

/// 单次连接的生命周期。返回一个人类可读的结束原因字符串。
async fn run_session(spec: &dyn ExchangeSpec, cfg: &Config) -> Result<String> {
    let frame_cap = spec.max_frame_size();
    let ws_cfg = WebSocketConfig::default()
        .max_frame_size(Some(frame_cap))
        .max_message_size(Some(frame_cap));
    let (mut ws, _) =
        tokio_tungstenite::connect_async_with_config(spec.ws_url(), Some(ws_cfg), true).await?;
    tracing::info!("{} connected to {}", spec.name(), spec.ws_url());

    for msg in spec.subscribe_msgs() {
        ws.send(Message::Text(msg.into())).await?;
    }

    let mut sink = Sink::new(cfg);
    let stale    = Duration::from_secs(cfg.ex_stale_s.max(1));

    // 两个定时器首次都会立即触发；跳过第一个 tick，
    // 这样失活时钟不会在我们还没有机会接收消息之前就开始计时。
    let mut ping_tick  = tokio::time::interval(Duration::from_secs(cfg.ex_ping_s.max(1)));
    let mut flush_tick = tokio::time::interval(Duration::from_secs(1));
    ping_tick.tick().await;
    flush_tick.tick().await;

    let mut last_msg = Instant::now();
    let mut msgs:  u64 = 0;
    let mut secs:  u64 = 0;

    let reason = loop {
        tokio::select! {
            frame = ws.next() => {
                let Some(frame) = frame else { break "stream closed".into() };
                match frame? {
                    Message::Text(t) => {
                        last_msg = Instant::now();
                        msgs += 1;
                        ingest(spec.parse(t.as_str(), now_ms()), &mut sink, cfg);
                        sink.maybe_flush().await;
                    }
                    Message::Binary(_)
                    | Message::Ping(_)
                    | Message::Pong(_) => { last_msg = Instant::now(); }
                    Message::Close(c) => break format!("close frame: {c:?}"),
                    Message::Frame(_) => {}
                }
            }
            _ = ping_tick.tick() => {
                let r = match spec.ping_msg() {
                    Some(p) => ws.send(Message::Text(p.into())).await,
                    None    => ws.send(Message::Ping(Default::default())).await,
                };
                if let Err(e) = r { break format!("ping send failed: {e}"); }
            }
            _ = flush_tick.tick() => {
                if last_msg.elapsed() > stale {
                    break format!("stale: no frame for {}s", last_msg.elapsed().as_secs());
                }
                sink.flush_all().await;
                secs += 1;
                if secs % 60 == 0 {
                    tracing::info!("{}: {} msgs recv, {} rows ok, {} rows failed",
                        spec.name(), msgs, sink.ok_rows, sink.fail_rows);
                }
            }
        }
    };

    sink.flush_all().await;
    Ok(reason)
}

// ── 辅助函数 ──────────────────────────────────────────────────────────────────

fn ingest(records: Vec<ParsedRecord>, sink: &mut Sink, cfg: &Config) {
    for rec in records {
        match rec {
            ParsedRecord::Bbo(r)   => { if cfg.is_target(&r.symbol) { sink.push_bbo(&r);   } }
            ParsedRecord::Trade(r) => { if cfg.is_target(&r.symbol) { sink.push_trade(&r); } }
            ParsedRecord::Book(r)  => { if cfg.is_target(&r.symbol) { sink.push_book(&r);  } }
        }
    }
}

#[inline]
fn now_ms() -> i64 { chrono::Utc::now().timestamp_millis() }
