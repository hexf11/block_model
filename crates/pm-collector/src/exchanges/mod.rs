pub mod binance;
pub mod bybit;
pub mod coinbase;
pub mod kraken;
pub mod okx;
pub(crate) mod util;

use crate::config::Config;
use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use pm_features::book::OrderBook;
use pm_features::types::{BboRecord, BookRecord, TradeRecord};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use tokio_tungstenite::tungstenite::Message;

/// 完整盘口的重播周期。
///
/// bybit/kraken/coinbase 只在连接时发一次快照，之后全是增量。离线重建
/// 若只能回溯到连接点，一个跑了几小时的会话就意味着要顺序重放数万条
/// 增量才能还原某个时刻的盘口——训练时对每个窗口都做一遍不现实。
///
/// 每 5 分钟落一份完整盘口，重建成本就被压到一个周期以内。
/// 代价可忽略：5 家 × 5 symbol × 12 次/小时 = 300 行/小时。
const BOOK_REPLAY_S: u64 = 300;

/// 本地维护盘口时保留的档数。50 档足够覆盖深度分布类特征，
/// 又不会让每 5 分钟的快照行变得过大。
const BOOK_DEPTH: usize = 50;

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

    /// 订阅完成后拉取一次 L2 底图。
    ///
    /// 只有 binance 需要：它的 `@depth@100ms` 是纯增量流，设计上永不发送
    /// 快照，官方要求配 REST `/api/v3/depth` 取底图。没有底图，增量重放
    /// 不出盘口——这些行在离线训练时是死数据。
    ///
    /// 返回的记录会以 is_snapshot=1 写入，作为后续增量的重建起点。
    /// 每次重连都会重新调用，因为断线期间错过的增量无法补回，
    /// 旧底图对新的增量流不再成立。
    fn fetch_snapshots<'a>(&'a self)
        -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<BookRecord>> + Send + 'a>>
    {
        Box::pin(async { Vec::new() })
    }
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

    let sink = crate::sink::spawn_writer(cfg);
    let stale    = Duration::from_secs(cfg.ex_stale_s.max(1));

    // 本会话内按 symbol 维护的本地盘口，只用于周期性重播完整快照。
    // 会话结束即丢弃：重连之后错过的增量补不回来，旧盘口对新的增量流
    // 不再成立，必须从新的快照重新起步。
    let mut books: HashMap<String, (pm_features::types::Exchange, OrderBook)> = HashMap::new();

    // L2 底图：必须在订阅之后拉取，否则快照与增量之间会留下缺口。
    // 顺序是 订阅 → 拉快照 → 处理增量：这样先到的增量已经在 socket
    // 缓冲区里排队，快照的 lastUpdateId 之后的部分都能被覆盖到。
    for rec in spec.fetch_snapshots().await {
        if cfg.is_target(&rec.symbol) {
            tracing::info!("{} L2 底图 {} lastUpdateId={:?}",
                spec.name(), rec.symbol, rec.seq);
            track_book(&mut books, &rec);
            sink.push_book(&rec);
        }
    }

    // 三个定时器首次都会立即触发；跳过第一个 tick，
    // 这样失活时钟不会在我们还没有机会接收消息之前就开始计时，
    // 重播也不会在盘口还是空的时候先跑一轮。
    let mut ping_tick   = tokio::time::interval(Duration::from_secs(cfg.ex_ping_s.max(1)));
    let mut flush_tick  = tokio::time::interval(Duration::from_secs(1));
    let mut replay_tick = tokio::time::interval(Duration::from_secs(BOOK_REPLAY_S));
    ping_tick.tick().await;
    flush_tick.tick().await;
    replay_tick.tick().await;

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
                        // 只入队，不等 ClickHouse —— 见 sink.rs 顶部关于
                        // 时间戳塌陷的说明。now_ms() 必须是收帧的真实时刻。
                        ingest(spec.parse(t.as_str(), now_ms()), &sink, cfg, &mut books);
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
            _ = replay_tick.tick() => {
                let n = replay_books(&books, &sink);
                if n > 0 {
                    tracing::info!("{}: 重播完整盘口 {n} 份", spec.name());
                }
            }
            _ = flush_tick.tick() => {
                if last_msg.elapsed() > stale {
                    break format!("stale: no frame for {}s", last_msg.elapsed().as_secs());
                }
                secs += 1;
                if secs % 60 == 0 {
                    let dropped = sink.dropped_count();
                    tracing::info!("{}: {} msgs recv, {} dropped", spec.name(), msgs, dropped);
                    // 非零就是真丢数据 —— channel 满意味着 ClickHouse 持续跟不上
                    if dropped > 0 {
                        tracing::error!("{}: 已丢弃 {dropped} 条记录，写入端跟不上采集速度",
                            spec.name());
                    }
                }
            }
        }
    };

    sink.request_flush();
    Ok(reason)
}

// ── 辅助函数 ──────────────────────────────────────────────────────────────────

fn ingest(
    records: Vec<ParsedRecord>,
    sink:    &crate::sink::WriterHandle,
    cfg:     &Config,
    books:   &mut HashMap<String, (pm_features::types::Exchange, OrderBook)>,
) {
    for rec in records {
        match rec {
            ParsedRecord::Bbo(r)   => { if cfg.is_target(&r.symbol) { sink.push_bbo(&r);   } }
            ParsedRecord::Trade(r) => { if cfg.is_target(&r.symbol) { sink.push_trade(&r); } }
            ParsedRecord::Book(r)  => {
                if cfg.is_target(&r.symbol) {
                    track_book(books, &r);
                    sink.push_book(&r);
                }
            }
        }
    }
}

/// 把一条盘口记录应用到本地盘口上。快照会重置，增量会叠加。
fn track_book(
    books: &mut HashMap<String, (pm_features::types::Exchange, OrderBook)>,
    rec:   &BookRecord,
) {
    books
        .entry(rec.symbol.clone())
        .or_insert_with(|| (rec.exchange.clone(), OrderBook::new(&rec.symbol, BOOK_DEPTH)))
        .1
        .apply(rec);
}

/// 把每个已就绪的本地盘口写成一条 is_snapshot=1 的记录。返回写出的份数。
///
/// 注意重播出来的快照只有 [`BOOK_DEPTH`] 档——它是本地盘口的镜像，不是
/// 交易所的完整深度。用途是给离线重建一个近端的起点，深端档位靠后续
/// 增量自行补齐。
fn replay_books(
    books: &HashMap<String, (pm_features::types::Exchange, OrderBook)>,
    sink:  &crate::sink::WriterHandle,
) -> usize {
    let now = now_ms();
    let mut n = 0;
    for (symbol, (exchange, book)) in books {
        if !book.is_ready() { continue; }
        let (bids, asks) = book.to_ladders();
        sink.push_book(&BookRecord {
            exchange:    exchange.clone(),
            symbol:      symbol.clone(),
            // 重播不是交易所事件，ts_ex 用本地时刻；离线时按 recv_ms
            // 对齐即可（泄漏边界本来就只认 recv_ms）。
            ts_ex:       now,
            recv_ms:     now,
            bids,
            asks,
            is_snapshot: true,
            // 沿用最后一条增量的序列号，重建时可据此判断该从哪条增量
            // 继续往下放。
            seq:         book.last_seq,
            first_seq:   book.last_seq,
        });
        n += 1;
    }
    n
}

#[inline]
fn now_ms() -> i64 { chrono::Utc::now().timestamp_millis() }
