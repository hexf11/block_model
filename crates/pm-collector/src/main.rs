// main.rs 从库里重新导出，这样二进制无需重复声明模块。
use pm_collector::config::Config;
use pm_collector::{rtds, exchanges};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env()
            .add_directive("pm_collector=info".parse()?))
        .init();

    let cfg = Config::from_env()?;
    tracing::info!("collector starting, target_symbols={:?}", cfg.target_symbols);

    tokio::try_join!(
        rtds::run(cfg.clone()),
        exchanges::run_all(cfg.clone()),
    )?;

    Ok(())
}
