//! 带 JSONL 持久化兜底的批量 ClickHouse 写入器。
//!
//! 数据量决定了兜底策略：
//!   * RTDS  —— 8 个 symbol，每秒 1 条。永远镜像到 JSONL：该数据源
//!     没有重放，JSONL 也是 ClickHouse 有意丢弃的非目标 symbol 的唯一留底。
//!     每天约 2 MB。
//!   * 交易所 —— 约 2500 msg/s。全量镜像会烧掉约 43 GB/天，所以
//!     只在 ClickHouse 插入失败时才写 JSONL。这样既保住"不丢数据"
//!     的保证，又不产生磁盘账单。
//!
//! ## 为什么写入必须离开读循环
//!
//! 早期版本在 WebSocket 读循环里直接 `sink.maybe_flush().await`。数据没丢
//! （缓冲无界、失败会 spill），但**时间戳塌了**：flush 等待 ClickHouse 的
//! 几毫秒里，WS 帧堆在 OS socket 缓冲区；插入返回后代码连读一整批，
//! `now_ms()` 在这期间几乎不变，于是整批报价拿到同一个 recv_ms。
//!
//! 实测证据——binance BTC 的 BBO 里，恰好 200 条挤在同一毫秒的情况出现了
//! 44 次，199/198/197 条各数次。200 正是 CH_BATCH_SIZE，这是 flush 阻塞
//! 留下的指纹。
//!
//! 后果对 OFI（order flow imbalance）是致命的：OFI 的定义就是逐笔比较相邻
//! 两条报价，时间戳塌成一个点之后，任何按时间窗口切分的 OFI 都会在边界
//! 错配。更要紧的是整个快照的防泄漏保证建立在 recv_ms 上——它必须是
//! "我们真正收到这一帧的时刻"，不能是"我们腾出手来处理它的时刻"。
//!
//! 所以现在：读循环只 `push_*`（纯内存操作，纳秒级），由独立 task negotiate
//! ClickHouse。`WriterHandle` 是读循环唯一接触的东西。

use crate::config::Config;
use clickhouse::{Client, Row, RowOwned, RowWrite};
use pm_features::types::{BboRecord, BookRecord, TradeRecord, TradeSide, TwapRecord};
use serde::Serialize;

// ── 与 ClickHouse schema 对应的行类型 ───────────────────────────────────────
// MATERIALIZED 列（ts、obs_ts、lag_ms）由服务端计算，不应出现在这里。

#[derive(Row, Serialize)]
pub struct BboRow {
    pub exchange: String,
    pub symbol:   String,
    pub ts_ex:    u64,
    pub recv_ms:  u64,
    pub bid:      String,
    pub bid_qty:  String,
    pub ask:      String,
    pub ask_qty:  String,
}

#[derive(Row, Serialize)]
pub struct TradeRow {
    pub exchange: String,
    pub symbol:   String,
    pub ts_ex:    u64,
    pub recv_ms:  u64,
    pub price:    String,
    pub qty:      String,
    pub side:     String,
    pub trade_id: String,
}

#[derive(Row, Serialize)]
pub struct BookRow {
    pub exchange:    String,
    pub symbol:      String,
    pub ts_ex:       u64,
    pub recv_ms:     u64,
    pub bids_json:   String,
    pub asks_json:   String,
    pub is_snapshot: u8,
    pub seq:         u64,
    pub first_seq:   u64,
}

#[derive(Row, Serialize)]
pub struct TwapRow {
    pub symbol:    String,
    pub window_s:  u16,
    pub obs_ms:    u64,
    pub pub_ms:    u64,
    pub recv_ms:   u64,
    pub value_e18: String,
    /// Decimal128(18)。缩放后的整数就是 `value_e18`，因此 Chainlink 原始
    /// 值可以无精度损失地往返存储。
    pub price:     i128,
}

// ── 转换 ────────────────────────────────────────────────────────────────────

fn side_str(s: TradeSide) -> &'static str {
    match s { TradeSide::Buy => "buy", TradeSide::Sell => "sell" }
}

/// 将价格档位序列化为 schema 期望的紧凑格式 `[[price, qty], ...]` JSON，
/// 完整保留十进制文本。
fn ladder_json(levels: &[[rust_decimal::Decimal; 2]]) -> String {
    let mut s = String::from("[");
    for (i, [p, q]) in levels.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&format!("[\"{p}\",\"{q}\"]"));
    }
    s.push(']');
    s
}

impl From<&BboRecord> for BboRow {
    fn from(r: &BboRecord) -> Self {
        Self {
            exchange: r.exchange.as_str().to_owned(),
            symbol:   r.symbol.clone(),
            ts_ex:    r.ts_ex.max(0) as u64,
            recv_ms:  r.recv_ms.max(0) as u64,
            bid:      r.bid.to_string(),
            bid_qty:  r.bid_qty.to_string(),
            ask:      r.ask.to_string(),
            ask_qty:  r.ask_qty.to_string(),
        }
    }
}

impl From<&TradeRecord> for TradeRow {
    fn from(r: &TradeRecord) -> Self {
        Self {
            exchange: r.exchange.as_str().to_owned(),
            symbol:   r.symbol.clone(),
            ts_ex:    r.ts_ex.max(0) as u64,
            recv_ms:  r.recv_ms.max(0) as u64,
            price:    r.price.to_string(),
            qty:      r.qty.to_string(),
            side:     side_str(r.side).to_owned(),
            trade_id: r.trade_id.clone(),
        }
    }
}

impl From<&BookRecord> for BookRow {
    fn from(r: &BookRecord) -> Self {
        Self {
            exchange:    r.exchange.as_str().to_owned(),
            symbol:      r.symbol.clone(),
            ts_ex:       r.ts_ex.max(0) as u64,
            recv_ms:     r.recv_ms.max(0) as u64,
            bids_json:   ladder_json(&r.bids),
            asks_json:   ladder_json(&r.asks),
            is_snapshot: r.is_snapshot as u8,
            seq:         r.seq.unwrap_or(0),
            first_seq:   r.first_seq.unwrap_or(0),
        }
    }
}

impl TryFrom<&TwapRecord> for TwapRow {
    type Error = std::num::ParseIntError;
    fn try_from(r: &TwapRecord) -> Result<Self, Self::Error> {
        Ok(Self {
            symbol:    r.symbol.clone(),
            window_s:  r.window_s,
            obs_ms:    r.obs_ms.max(0) as u64,
            pub_ms:    r.pub_ms.max(0) as u64,
            recv_ms:   r.recv_ms.max(0) as u64,
            value_e18: r.value_e18.clone(),
            price:     r.value_e18.parse::<i128>()?,
        })
    }
}

// ── 批量写入 Sink ───────────────────────────────────────────────────────────

/// 按表缓冲行，达到 `batch_size` 后刷新各缓冲区。
/// 插入失败时将该批次追加到 JSONL 以便后续重放，而不是静默丢弃。
pub struct Sink {
    client:     Client,
    batch_size: usize,
    backup_dir: std::path::PathBuf,

    bbo:    Vec<BboRow>,
    trades: Vec<TradeRow>,
    book:   Vec<BookRow>,
    twap:   Vec<TwapRow>,

    pub ok_rows:   u64,
    pub fail_rows: u64,
}

impl Sink {
    pub fn new(cfg: &Config) -> Self {
        // clickhouse crate 使用 HTTP 协议；.env 里的 CH_PORT 是
        // native 协议端口（9000），所以映射到 HTTP 端口（8123），
        // 除非运维人员直接配置的就是 HTTP 端口。
        let http_port = if cfg.ch_port == 9000 { 8123 } else { cfg.ch_port };
        let client = Client::default()
            .with_url(format!("http://{}:{}", cfg.ch_host, http_port))
            .with_user(&cfg.ch_user)
            .with_password(&cfg.ch_password)
            .with_database(&cfg.ch_database);

        Self {
            client,
            batch_size: cfg.ch_batch_sz,
            backup_dir: std::path::PathBuf::from(&cfg.backup_dir),
            bbo:    Vec::new(),
            trades: Vec::new(),
            book:   Vec::new(),
            twap:   Vec::new(),
            ok_rows:   0,
            fail_rows: 0,
        }
    }

    pub fn push_bbo(&mut self, r: &BboRecord)     { self.bbo.push(r.into()); }
    pub fn push_trade(&mut self, r: &TradeRecord) { self.trades.push(r.into()); }
    pub fn push_book(&mut self, r: &BookRecord)   { self.book.push(r.into()); }

    /// 当 value_e18 不是有效的 i128 时返回 false，
    /// 以便调用方计数丢弃，而不是悄悄写入错误价格。
    pub fn push_twap(&mut self, r: &TwapRecord) -> bool {
        match TwapRow::try_from(r) {
            Ok(row) => { self.twap.push(row); true }
            Err(e) => {
                tracing::warn!("twap value_e18 无法解析 ({e})，丢弃: {}", r.value_e18);
                false
            }
        }
    }

    /// 刷新所有达到批量阈值的缓冲区。
    pub async fn maybe_flush(&mut self) {
        if self.bbo.len()    >= self.batch_size { self.flush_bbo().await; }
        if self.trades.len() >= self.batch_size { self.flush_trades().await; }
        if self.book.len()   >= self.batch_size { self.flush_book().await; }
        if self.twap.len()   >= self.batch_size { self.flush_twap().await; }
    }

    /// 不论阈值如何，刷新全部缓冲区。
    /// 在关闭和断线时调用，避免半批数据滞留在内存里。
    pub async fn flush_all(&mut self) {
        self.flush_bbo().await;
        self.flush_trades().await;
        self.flush_book().await;
        self.flush_twap().await;
    }

    async fn flush_bbo(&mut self)    { let b = std::mem::take(&mut self.bbo);    self.write(b, "ex_bbo").await; }
    async fn flush_trades(&mut self) { let b = std::mem::take(&mut self.trades); self.write(b, "ex_trades").await; }
    async fn flush_book(&mut self)   { let b = std::mem::take(&mut self.book);   self.write(b, "ex_book").await; }
    async fn flush_twap(&mut self)   { let b = std::mem::take(&mut self.twap);   self.write(b, "twap").await; }

    /// 插入一批数据。任何失败都会把这批行追加到
    /// `<backup_dir>/<table>.failed.jsonl`，保证不会静默丢弃。
    async fn write<T>(&mut self, rows: Vec<T>, table: &str)
    where
        T: RowOwned + RowWrite + Serialize,
    {
        if rows.is_empty() { return; }
        let n = rows.len() as u64;

        match self.insert_batch(&rows, table).await {
            Ok(()) => self.ok_rows += n,
            Err(e) => {
                self.fail_rows += n;
                tracing::error!("插入 {table} 失败（{n} 行）: {e}");
                self.spill(&rows, table);
            }
        }
    }

    async fn insert_batch<T>(&self, rows: &[T], table: &str) -> anyhow::Result<()>
    where
        T: RowOwned + RowWrite,
    {
        let mut insert = self.client.insert::<T>(table).await?;
        for r in rows {
            insert.write(r).await?;
        }
        insert.end().await?;
        Ok(())
    }

    /// 将失败的批次追加到 JSONL，刻意使用阻塞 IO：
    /// 走到这条路径说明 ClickHouse 已经挂了，此时持久性比延迟更重要。
    fn spill<T: Serialize>(&self, rows: &[T], table: &str) {
        use std::io::Write;
        if let Err(e) = std::fs::create_dir_all(&self.backup_dir) {
            tracing::error!("无法创建备份目录: {e}");
            return;
        }
        let path = self.backup_dir.join(format!("{table}.failed.jsonl"));
        let file = std::fs::OpenOptions::new().create(true).append(true).open(&path);
        let Ok(mut f) = file else {
            tracing::error!("无法打开 {}: {:?}", path.display(), file.err());
            return;
        };
        let mut buf = String::new();
        for r in rows {
            match serde_json::to_string(r) {
                Ok(line) => { buf.push_str(&line); buf.push('\n'); }
                Err(e) => tracing::error!("spill 序列化失败: {e}"),
            }
        }
        if let Err(e) = f.write_all(buf.as_bytes()) {
            tracing::error!("spill 写入失败: {e}");
        }
    }

    /// 无条件将记录镜像到 JSONL。用于 RTDS——该数据源无法重放，数据量也微不足道。
    pub fn mirror_jsonl<T: Serialize>(&self, rec: &T, name: &str) {
        use std::io::Write;
        if std::fs::create_dir_all(&self.backup_dir).is_err() { return; }
        let day = chrono::Utc::now().format("%Y%m%d");
        let path = self.backup_dir.join(format!("{name}_{day}.jsonl"));
        let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) else { return };
        if let Ok(line) = serde_json::to_string(rec) {
            let _ = writeln!(f, "{line}");
        }
    }
}

// ── 异步写入器 ──────────────────────────────────────────────────────────────

/// 一批待写入的记录。读循环把这些丢进 channel 就立刻返回。
pub enum WriteMsg {
    Bbo(BboRow),
    Trade(TradeRow),
    Book(BookRow),
    Twap(TwapRow),
    /// 请求立即刷盘（断线时用，确保半批数据不滞留）
    Flush,
}

/// 读循环持有的写入句柄。所有方法都是非阻塞的纯内存操作。
#[derive(Clone)]
pub struct WriterHandle {
    tx: tokio::sync::mpsc::Sender<WriteMsg>,
    /// channel 满时丢弃的记录数。非零就说明写入端跟不上采集速度，
    /// 必须调查——这是唯一会真正丢数据的路径。
    pub dropped: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl WriterHandle {
    /// 入队一条记录。channel 满时计数并丢弃，绝不阻塞读循环——
    /// 阻塞会重新引入时间戳塌陷问题，那比丢几条记录更糟：
    /// 丢记录是可观测的（dropped 计数），时间戳塌陷是静默的。
    fn send(&self, msg: WriteMsg) {
        if self.tx.try_send(msg).is_err() {
            self.dropped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    pub fn push_bbo(&self, r: &BboRecord)     { self.send(WriteMsg::Bbo(r.into())); }
    pub fn push_trade(&self, r: &TradeRecord) { self.send(WriteMsg::Trade(r.into())); }
    pub fn push_book(&self, r: &BookRecord)   { self.send(WriteMsg::Book(r.into())); }

    pub fn push_twap(&self, r: &TwapRecord) -> bool {
        match TwapRow::try_from(r) {
            Ok(row) => { self.send(WriteMsg::Twap(row)); true }
            Err(e) => {
                tracing::warn!("twap value_e18 无法解析 ({e})，丢弃: {}", r.value_e18);
                false
            }
        }
    }

    pub fn request_flush(&self) { self.send(WriteMsg::Flush); }

    pub fn dropped_count(&self) -> u64 {
        self.dropped.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// 启动写入 task，返回读循环用的句柄。
///
/// channel 容量按最坏情况定：binance BTC 的 BBO 单秒峰值实测 1407 条，
/// 五个 symbol 五家交易所叠加后短时可达数千。65536 给了约 10 秒的缓冲，
/// 足以吸收 ClickHouse 的偶发慢插入，又不至于在真正故障时无限吃内存。
pub fn spawn_writer(cfg: &Config) -> WriterHandle {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<WriteMsg>(65536);
    let dropped = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let mut sink = Sink::new(cfg);

    tokio::spawn(async move {
        // 兜底定时刷盘：低频数据（kraken 0.24/s）不该在缓冲区里等到攒够一批
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(1));
        tick.tick().await;
        loop {
            tokio::select! {
                msg = rx.recv() => {
                    let Some(msg) = msg else { break };   // 所有 handle 已释放
                    match msg {
                        WriteMsg::Bbo(r)   => sink.bbo.push(r),
                        WriteMsg::Trade(r) => sink.trades.push(r),
                        WriteMsg::Book(r)  => sink.book.push(r),
                        WriteMsg::Twap(r)  => sink.twap.push(r),
                        WriteMsg::Flush    => sink.flush_all().await,
                    }
                    sink.maybe_flush().await;
                }
                _ = tick.tick() => { sink.flush_all().await; }
            }
        }
        sink.flush_all().await;
    });

    WriterHandle { tx, dropped }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pm_features::types::Exchange;
    use rust_decimal::Decimal;
    use std::str::FromStr;

    fn twap(value_e18: &str) -> TwapRecord {
        TwapRecord {
            symbol:    "btc/usd".into(),
            window_s:  30,
            obs_ms:    1_754_899_999_000,
            pub_ms:    1_754_899_999_500,
            recv_ms:   1_754_900_000_000,
            value_e18: value_e18.into(),
            price:     Decimal::ZERO, // 行转换不使用此字段
        }
    }

    #[test]
    fn twap_price_is_the_raw_scaled_integer() {
        // Decimal128(18) 存储缩放后的整数，Chainlink 给的就是这个——
        // 无需除法，无精度损失。
        let row = TwapRow::try_from(&twap("112345670000000000000000")).unwrap();
        assert_eq!(row.price, 112_345_670_000_000_000_000_000i128);
        assert_eq!(row.value_e18, "112345670000000000000000");
    }

    #[test]
    fn twap_handles_values_beyond_u64() {
        // 六位数的 BTC 价格乘以 1e18 会溢出 u64（上限约 1.8e19），
        // 所以 i128 列是真正必需的，不是防御性设计。
        let v = "112345670000000000000000";
        assert!(v.parse::<u64>().is_err(), "值必须超出 u64 范围");
        assert!(TwapRow::try_from(&twap(v)).is_ok());
    }

    #[test]
    fn twap_rejects_unparseable_value() {
        assert!(TwapRow::try_from(&twap("1.5e18")).is_err());
        assert!(TwapRow::try_from(&twap("")).is_err());
    }

    #[test]
    fn ladder_json_preserves_decimal_text() {
        let levels = [
            [Decimal::from_str("0.0024").unwrap(), Decimal::from_str("10").unwrap()],
            [Decimal::from_str("0.0023").unwrap(), Decimal::ZERO],
        ];
        assert_eq!(ladder_json(&levels), r#"[["0.0024","10"],["0.0023","0"]]"#);
        assert_eq!(ladder_json(&[]), "[]");
    }

    #[test]
    fn bbo_row_normalises_negative_timestamps() {
        // 时钟偏移或交易所字段异常不应导致包装成超大 UInt64。
        let r = BboRecord {
            exchange: Exchange::Binance,
            symbol:   "btc/usd".into(),
            ts_ex:    -1,
            recv_ms:  1_754_900_000_000,
            bid:      Decimal::from_str("111999.5").unwrap(),
            bid_qty:  Decimal::from_str("2.1").unwrap(),
            ask:      Decimal::from_str("112000.5").unwrap(),
            ask_qty:  Decimal::from_str("0.9").unwrap(),
        };
        let row = BboRow::from(&r);
        assert_eq!(row.ts_ex, 0);
        assert_eq!(row.exchange, "binance");
        assert_eq!(row.bid, "111999.5");
    }

    #[test]
    fn book_row_flattens_snapshot_flag_and_missing_seq() {
        let r = BookRecord {
            exchange:    Exchange::Okx,
            symbol:      "eth/usd".into(),
            ts_ex:       1,
            recv_ms:     2,
            bids:        vec![[Decimal::from_str("3011.4").unwrap(), Decimal::from_str("3").unwrap()]],
            asks:        vec![],
            is_snapshot: true,
            seq:         None,
            first_seq:   None,
        };
        let row = BookRow::from(&r);
        assert_eq!(row.is_snapshot, 1);
        assert_eq!(row.seq, 0, "缺失的 seq 应变为 0，而不是包装后的哨兵值");
        assert_eq!(row.first_seq, 0);
        assert_eq!(row.asks_json, "[]");
    }

    #[test]
    fn trade_side_serialises_lowercase() {
        assert_eq!(side_str(TradeSide::Buy), "buy");
        assert_eq!(side_str(TradeSide::Sell), "sell");
    }
}
