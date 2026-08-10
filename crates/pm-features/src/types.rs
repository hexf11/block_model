use serde::{Deserialize, Serialize};
use rust_decimal::Decimal;

/// 全项目通用的规范 symbol 格式，例如 "btc/usd"
pub type Symbol = String;

/// 交易所标识
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Exchange {
    Binance,
    Okx,
    Bybit,
    Coinbase,
    Kraken,
}

impl Exchange {
    pub fn as_str(&self) -> &'static str {
        match self {
            Exchange::Binance  => "binance",
            Exchange::Okx      => "okx",
            Exchange::Bybit    => "bybit",
            Exchange::Coinbase => "coinbase",
            Exchange::Kraken   => "kraken",
        }
    }
}

/// 来自单一交易所的最优买价 / 最优卖价快照
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BboRecord {
    pub exchange: Exchange,
    pub symbol:   Symbol,
    pub ts_ex:    i64,       // 交易所时间戳，毫秒
    pub recv_ms:  i64,       // 本地接收时间戳，毫秒
    pub bid:      Decimal,
    pub bid_qty:  Decimal,
    pub ask:      Decimal,
    pub ask_qty:  Decimal,
}

impl BboRecord {
    pub fn mid(&self) -> Decimal {
        (self.bid + self.ask) / Decimal::TWO
    }

    pub fn spread(&self) -> Decimal {
        self.ask - self.bid
    }

    /// bid_qty / (bid_qty + ask_qty) —— 订单流失衡信号
    pub fn imbalance(&self) -> Option<Decimal> {
        let total = self.bid_qty + self.ask_qty;
        if total.is_zero() { None } else { Some(self.bid_qty / total) }
    }
}

/// 来自单一交易所的一笔成交
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub exchange: Exchange,
    pub symbol:   Symbol,
    pub ts_ex:    i64,
    pub recv_ms:  i64,
    pub price:    Decimal,
    pub qty:      Decimal,
    pub side:     TradeSide,
    pub trade_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TradeSide {
    Buy,
    Sell,
}

/// L2 订单簿更新（快照或增量）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookRecord {
    pub exchange:    Exchange,
    pub symbol:      Symbol,
    pub ts_ex:       i64,
    pub recv_ms:     i64,
    /// [price, qty] 对 —— qty=0 表示删除该档位
    pub bids:        Vec<[Decimal; 2]>,
    pub asks:        Vec<[Decimal; 2]>,
    pub is_snapshot: bool,
    pub seq:         Option<u64>,
}

/// Polymarket RTDS twap30 数据点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwapRecord {
    pub symbol:     Symbol,
    pub window_s:   u16,
    pub obs_ms:     i64,    // Chainlink 观测时间
    pub pub_ms:     i64,    // RTDS 发布时间
    pub recv_ms:    i64,    // 本地接收时间
    pub value_e18:  String, // 来自 Chainlink 的原始 E18 字符串
    pub price:      Decimal,
}

/// 预测时刻（T-110s 快照）计算得到的特征向量
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    pub symbol:     Symbol,
    pub snap_ms:    i64,            // 本次快照的时间戳
    pub features:   Vec<f64>,       // 按顺序排列、喂给模型的特征值
    pub feature_names: Vec<String>, // 与上面同序，用于日志 / 调试
}
