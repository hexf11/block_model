mod config;

use axum::{routing::post, Router, Json};
use serde::{Deserialize, Serialize};
use tracing_subscriber::EnvFilter;

#[derive(Deserialize)]
struct PredictRequest {
    symbol:   String,
    snap_ms:  i64,
    features: Vec<f64>,
}

#[derive(Serialize)]
struct PredictResponse {
    symbol:    String,
    snap_ms:   i64,
    up_prob:   f64,
    down_prob: f64,
}

async fn predict(Json(req): Json<PredictRequest>) -> Json<PredictResponse> {
    // TODO: 加载 ONNX 模型并运行推理
    Json(PredictResponse {
        symbol:    req.symbol,
        snap_ms:   req.snap_ms,
        up_prob:   0.5,
        down_prob: 0.5,
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env()
            .add_directive("pm_inference=info".parse()?))
        .init();

    let cfg = config::Config::from_env()?;
    let addr = format!("{}:{}", cfg.bind_host, cfg.bind_port);

    let app = Router::new()
        .route("/predict", post(predict))
        .route("/health",  axum::routing::get(|| async { "ok" }));

    tracing::info!("inference service listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
