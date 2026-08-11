use crate::types::BookRecord;
use rust_decimal::Decimal;
use std::collections::BTreeMap;

/// 依据快照 + 增量流维护一份本地 L2 订单簿。
/// 只保留前 N 档以限制内存占用。
pub struct OrderBook {
    pub symbol: String,
    pub max_levels: usize,
    /// price -> qty，按价格升序
    bids: BTreeMap<Decimal, Decimal>,
    asks: BTreeMap<Decimal, Decimal>,
    pub last_seq: Option<u64>,
    pub last_ts: i64,
}

impl OrderBook {
    pub fn new(symbol: impl Into<String>, max_levels: usize) -> Self {
        Self {
            symbol: symbol.into(),
            max_levels,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_seq: None,
            last_ts: 0,
        }
    }

    pub fn apply(&mut self, rec: &BookRecord) {
        if rec.is_snapshot {
            self.bids.clear();
            self.asks.clear();
        }
        for [price, qty] in &rec.bids {
            if qty.is_zero() { self.bids.remove(price); } else { self.bids.insert(*price, *qty); }
        }
        for [price, qty] in &rec.asks {
            if qty.is_zero() { self.asks.remove(price); } else { self.asks.insert(*price, *qty); }
        }
        self.trim();
        self.last_seq = rec.seq;
        self.last_ts  = rec.ts_ex;
    }

    /// 买卖两侧各只保留前 `max_levels` 档
    fn trim(&mut self) {
        while self.bids.len() > self.max_levels {
            if let Some((&k, _)) = self.bids.iter().next() { self.bids.remove(&k); } else { break; }
        }
        while self.asks.len() > self.max_levels {
            if let Some((&k, _)) = self.asks.iter().next_back() { self.asks.remove(&k); } else { break; }
        }
    }

    pub fn best_bid(&self) -> Option<(Decimal, Decimal)> {
        self.bids.iter().next_back().map(|(&p, &q)| (p, q))
    }

    pub fn best_ask(&self) -> Option<(Decimal, Decimal)> {
        self.asks.iter().next().map(|(&p, &q)| (p, q))
    }

    /// 距最优买价 `range` 个价格单位以内的买单总量
    pub fn depth_bid(&self, range: Decimal) -> f64 {
        let Some((best, _)) = self.best_bid() else { return 0.0 };
        let floor = best - range;
        self.bids.iter()
            .filter(|(&p, _)| p >= floor)
            .map(|(_, &q)| q.try_into().unwrap_or(0.0))
            .sum()
    }

    pub fn depth_ask(&self, range: Decimal) -> f64 {
        let Some((best, _)) = self.best_ask() else { return 0.0 };
        let ceil = best + range;
        self.asks.iter()
            .filter(|(&p, _)| p <= ceil)
            .map(|(_, &q)| q.try_into().unwrap_or(0.0))
            .sum()
    }

    pub fn level_count(&self) -> (usize, usize) {
        (self.bids.len(), self.asks.len())
    }

    /// 导出当前盘口的全部档位，买盘按价格降序、卖盘按升序（即从最优档往外）。
    ///
    /// 用于周期性快照重播：bybit/kraken/coinbase 只在连接时发一次快照，
    /// 之后全是增量。离线重建若只能回溯到连接点，就得顺序重放数万条增量；
    /// 定期落一份完整盘口，把重建成本压到一个重播周期以内。
    pub fn to_ladders(&self) -> (Vec<[Decimal; 2]>, Vec<[Decimal; 2]>) {
        let bids = self.bids.iter().rev().map(|(&p, &q)| [p, q]).collect();
        let asks = self.asks.iter().map(|(&p, &q)| [p, q]).collect();
        (bids, asks)
    }

    /// 盘口是否已就绪（两侧都有档位）。未就绪时重播快照没有意义。
    pub fn is_ready(&self) -> bool {
        !self.bids.is_empty() && !self.asks.is_empty()
    }
}
