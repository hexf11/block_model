use anyhow::Result;

#[derive(Debug, Clone)]
pub struct Config {
    pub bind_host: String,
    pub bind_port:  u16,
    pub model_path: String,
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
            model_path:  env_str("MODEL_PATH",     "./models/stage_b.onnx"),
            ch_host:     env_str("CH_HOST",        "localhost"),
            ch_port:     env_u16("CH_PORT",        9000),
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
