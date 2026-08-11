use std::path::PathBuf;

use anyhow::Result;

#[derive(Debug, Clone)]
pub struct Config {
    pub bind_host: String,
    pub bind_port:  u16,
    pub model_path: PathBuf,
    pub calibration_path: PathBuf,
    pub ch_host:    String,
    pub ch_port:    u16,
    pub ch_user:    String,
    pub ch_password: String,
    pub ch_database: String,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();
        Ok(Self {
            bind_host:   env_str("INFERENCE_HOST", "127.0.0.1"),
            bind_port:   env_u16("INFERENCE_PORT", 8765),
            model_path:  PathBuf::from(env_str("MODEL_PATH", "./models/model.onnx")),
            calibration_path: PathBuf::from(env_str("CALIBRATION_PATH", "./models/calibration.json")),
            ch_host:     env_str("CH_HOST",        "localhost"),
            // 与 pm-collector 相同的约定：.env 里的 CH_PORT 是 native 协议端口（9000），
            // HTTP 消费者映射到 8123。
            ch_port:     {
                let p = env_u16("CH_PORT", 9000);
                if p == 9000 { 8123 } else { p }
            },
            ch_user:     env_str("CH_USER",        "default"),
            ch_password: env_str("CH_PASSWORD",    ""),
            ch_database: env_str("CH_DATABASE",    "pm"),
        })
    }
}

fn env_str(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_owned())
}
fn env_u16(key: &str, default: u16) -> u16 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
