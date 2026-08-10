//! 交易所适配器的共享解析工具函数。

use pm_features::types::TradeSide;
use rust_decimal::Decimal;
use serde_json::Value;
use std::str::FromStr;

/// 将 JSON 值（可能是带引号的字符串或裸数字）解析为 Decimal。
/// 交易所以字符串发送价格以保留精度；我们保留这种精度。
pub fn dec(v: &Value) -> Option<Decimal> {
    match v {
        Value::String(s) => Decimal::from_str(s).ok(),
        Value::Number(n) => Decimal::from_str(&n.to_string()).ok(),
        _ => None,
    }
}

/// 与 `dec` 相同，但在解析失败时返回零，适用于数量字段——
/// 缺失值在语义上等同于"无数量"。
pub fn dec_or_zero(v: &Value) -> Decimal {
    dec(v).unwrap_or(Decimal::ZERO)
}

/// 从 `[[price, qty], ...]` 格式的档位数组中取出第一层，返回 (price, qty)。
pub fn top_level(ladder: Option<&Value>) -> Option<(Decimal, Decimal)> {
    let arr = ladder?.as_array()?;
    let first = arr.first()?.as_array()?;
    Some((dec(first.first()?)?, dec_or_zero(first.get(1)?)))
}

/// 将 `[[price, qty], ...]` JSON 档位数组转换为 Decimal 对的 Vec。
/// 解析失败的档位会被丢弃，而不是污染整个批次。
pub fn ladder(v: Option<&Value>) -> Vec<[Decimal; 2]> {
    let Some(arr) = v.and_then(|x| x.as_array()) else { return Vec::new() };
    arr.iter()
        .filter_map(|lvl| {
            let l = lvl.as_array()?;
            Some([dec(l.first()?)?, dec_or_zero(l.get(1)?)])
        })
        .collect()
}

/// 时间戳，可能以数字或数字字符串形式到达。
pub fn ts(v: Option<&Value>, fallback: i64) -> i64 {
    match v {
        Some(Value::Number(n)) => n.as_i64().unwrap_or(fallback),
        Some(Value::String(s)) => s.parse().unwrap_or(fallback),
        _ => fallback,
    }
}

/// ISO-8601 / RFC-3339 时间戳转为 epoch 毫秒（Coinbase、Kraken 使用）。
pub fn iso_ms(v: Option<&Value>, fallback: i64) -> i64 {
    v.and_then(|x| x.as_str())
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|d| d.timestamp_millis())
        .unwrap_or(fallback)
}

/// 字符串字段转为 owned String，缺失时返回空字符串。
pub fn s(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(x)) => x.clone(),
        Some(Value::Number(n)) => n.to_string(),
        _ => String::new(),
    }
}

/// 统一各交易所对 taker 方向的多种写法。
/// 对不识别的值返回 None，以便丢弃记录而不是悄悄打上错误标签——
/// taker 方向是模型特征，错误的值比缺失的行危害更大。
pub fn side(raw: &str) -> Option<TradeSide> {
    match raw.to_ascii_lowercase().as_str() {
        "buy" | "b" | "bid" => Some(TradeSide::Buy),
        "sell" | "s" | "ask" | "offer" => Some(TradeSide::Sell),
        _ => None,
    }
}

/// 从交易所原生 id 查找规范化的 `btc/usd` 格式 symbol。
pub fn sym_lookup(table: &[(&'static str, &'static str)], key: &str) -> Option<&'static str> {
    table.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}
