use crate::types::BboRecord;
use std::collections::VecDeque;

/// 在滑动窗口内的 BBO 数据点上计算滚动特征
pub struct BboWindow {
    pub symbol: String,
    ticks: VecDeque<BboRecord>,
    window_ms: i64,
}

impl BboWindow {
    pub fn new(symbol: impl Into<String>, window_ms: i64) -> Self {
        Self { symbol: symbol.into(), ticks: VecDeque::new(), window_ms }
    }

    pub fn push(&mut self, rec: BboRecord) {
        self.ticks.push_back(rec);
        self.evict();
    }

    fn evict(&mut self) {
        if let Some(newest) = self.ticks.back().map(|r| r.recv_ms) {
            while self.ticks.front().map_or(false, |r| newest - r.recv_ms > self.window_ms) {
                self.ticks.pop_front();
            }
        }
    }

    pub fn latest(&self) -> Option<&BboRecord> {
        self.ticks.back()
    }

    /// 窗口内的中间价均值
    pub fn mean_mid(&self) -> Option<f64> {
        if self.ticks.is_empty() { return None; }
        let sum: f64 = self.ticks.iter()
            .map(|r| r.mid().try_into().unwrap_or(f64::NAN))
            .sum();
        Some(sum / self.ticks.len() as f64)
    }

    /// 窗口内的平均订单簿失衡值
    pub fn mean_imbalance(&self) -> Option<f64> {
        let vals: Vec<f64> = self.ticks.iter()
            .filter_map(|r| r.imbalance())
            .map(|d| d.try_into().unwrap_or(f64::NAN))
            .collect();
        if vals.is_empty() { return None; }
        Some(vals.iter().sum::<f64>() / vals.len() as f64)
    }

    pub fn tick_count(&self) -> usize {
        self.ticks.len()
    }
}
