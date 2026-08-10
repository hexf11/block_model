use crate::types::{TradeRecord, TradeSide};
use std::collections::VecDeque;

/// 在滑动时间窗口内计算滚动订单流特征
pub struct TradeWindow {
    pub symbol: String,
    trades: VecDeque<TradeRecord>,
    window_ms: i64,
}

impl TradeWindow {
    pub fn new(symbol: impl Into<String>, window_ms: i64) -> Self {
        Self { symbol: symbol.into(), trades: VecDeque::new(), window_ms }
    }

    pub fn push(&mut self, rec: TradeRecord) {
        self.trades.push_back(rec);
        self.evict();
    }

    fn evict(&mut self) {
        if let Some(newest) = self.trades.back().map(|r| r.recv_ms) {
            while self.trades.front().map_or(false, |r| newest - r.recv_ms > self.window_ms) {
                self.trades.pop_front();
            }
        }
    }

    /// 净流量：买量 - 卖量（正值表示买方主导）
    pub fn net_flow(&self) -> f64 {
        self.trades.iter().fold(0f64, |acc, t| {
            let q: f64 = t.qty.try_into().unwrap_or(0.0);
            match t.side {
                TradeSide::Buy  => acc + q,
                TradeSide::Sell => acc - q,
            }
        })
    }

    pub fn buy_volume(&self) -> f64 {
        self.trades.iter()
            .filter(|t| t.side == TradeSide::Buy)
            .map(|t| t.qty.try_into().unwrap_or(0.0))
            .sum()
    }

    pub fn sell_volume(&self) -> f64 {
        self.trades.iter()
            .filter(|t| t.side == TradeSide::Sell)
            .map(|t| t.qty.try_into().unwrap_or(0.0))
            .sum()
    }

    pub fn trade_count(&self) -> usize {
        self.trades.len()
    }

    /// 窗口内所有成交的成交量加权平均价格
    pub fn vwap(&self) -> Option<f64> {
        let (pv, v) = self.trades.iter().fold((0f64, 0f64), |(pv, v), t| {
            let p: f64 = t.price.try_into().unwrap_or(0.0);
            let q: f64 = t.qty.try_into().unwrap_or(0.0);
            (pv + p * q, v + q)
        });
        if v == 0.0 { None } else { Some(pv / v) }
    }
}
