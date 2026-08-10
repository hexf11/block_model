use clickhouse::Client;
use crate::config::Config;

pub fn make_client(cfg: &Config) -> Client {
    // clickhouse crate 使用 HTTP 协议；将 native 端口 9000 映射到 8123，
    // 除非运维人员直接配置的就是 HTTP 端口。
    let port = if cfg.ch_port == 9000 { 8123 } else { cfg.ch_port };
    Client::default()
        .with_url(format!("http://{}:{}", cfg.ch_host, port))
        .with_user(cfg.ch_user.clone())
        .with_password(cfg.ch_password.clone())
        .with_database(cfg.ch_database.clone())
}

/// 将连接生命周期事件（connecting / disconnected / error）写入
/// `pm.conn_events`。写入失败只记录日志并吞掉，绝不打断调用方的重连循环。
pub async fn write_conn_event(client: &Client, source: &str, event: &str, detail: &str) {
    let now  = chrono::Utc::now().timestamp_millis();
    // 转义 detail 字符串中的单引号，防止内容任意的错误消息造成 SQL 注入。
    let safe = detail.replace('\'', "\\'");
    let sql  = format!(
        "INSERT INTO conn_events (ts_ms, source, event, detail) \
         VALUES ({now}, '{source}', '{event}', '{safe}')"
    );
    if let Err(e) = client.query(&sql).execute().await {
        tracing::warn!("conn_event({source}/{event}) write failed: {e}");
    }
}

