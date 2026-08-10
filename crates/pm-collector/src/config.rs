use anyhow::Result;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct Config {
    // ClickHouse
    pub ch_host:     String,
    pub ch_port:     u16,
    pub ch_user:     String,
    pub ch_password: String,
    pub ch_database: String,
    pub ch_batch_sz: usize,

    // RTDS
    pub rtds_url:          String,
    pub rtds_ping_s:       u64,
    pub rtds_stale_s:      u64,
    pub rtds_backoff_init: u64,
    pub rtds_backoff_max:  u64,

    // Exchange collectors
    pub ex_stale_s:       u64,
    pub ex_ping_s:        u64,
    pub ex_backoff_init:  u64,
    pub ex_backoff_max:   u64,

    // 写入 ClickHouse 的目标 symbol（JSONL 仍保存全量）
    pub target_symbols: HashSet<String>,

    // JSONL 备份目录
    pub backup_dir: String,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();

        let target_symbols = std::env::var("TARGET_SYMBOLS")
            .unwrap_or_else(|_| "btc/usd,eth/usd,sol/usd,xrp/usd,doge/usd".into())
            .split(',')
            .map(|s| s.trim().to_lowercase())
            .collect();

        Ok(Self {
            ch_host:     env_str("CH_HOST",     "localhost"),
            ch_port:     env_u16("CH_PORT",     9000),
            ch_user:     env_str("CH_USER",     "default"),
            ch_password: env_str("CH_PASSWORD", ""),
            ch_database: env_str("CH_DATABASE", "pm"),
            ch_batch_sz: env_usize("CH_BATCH_SIZE", 200),

            rtds_url:          env_str("RTDS_URL", "wss://ws-live-data.polymarket.com"),
            rtds_ping_s:       env_u64("RTDS_PING_INTERVAL",  5),
            rtds_stale_s:      env_u64("RTDS_STALE_TIMEOUT",  30),
            rtds_backoff_init: env_u64("RTDS_BACKOFF_INIT",   1),
            rtds_backoff_max:  env_u64("RTDS_BACKOFF_MAX",    60),

            ex_stale_s:      env_u64("EX_STALE_TIMEOUT", 20),
            ex_ping_s:       env_u64("EX_PING_INTERVAL", 15),
            ex_backoff_init: env_u64("EX_BACKOFF_INIT",  1),
            ex_backoff_max:  env_u64("EX_BACKOFF_MAX",   60),

            target_symbols,
            backup_dir: env_str("BACKUP_DIR", "./data/raw"),
        })
    }

    pub fn ch_url(&self) -> String {
        format!("tcp://{}:{}@{}:{}/{}",
            self.ch_user, self.ch_password,
            self.ch_host, self.ch_port,
            self.ch_database)
    }

    pub fn is_target(&self, symbol: &str) -> bool {
        self.target_symbols.contains(&symbol.to_lowercase())
    }
}

// ── 辅助函数 ────────────────────────────────────────────────────────────────
fn env_str(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_owned())
}
fn env_u16(key: &str, default: u16) -> u16 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
