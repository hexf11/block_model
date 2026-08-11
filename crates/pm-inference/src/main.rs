mod config;
mod model;

use std::sync::Arc;
use std::sync::Mutex;

use axum::extract::State;
use axum::routing::post;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tracing_subscriber::EnvFilter;

use pm_features::features::compute_features;
use pm_features::snapshot::SnapshotBuilder;

use crate::model::Model;

/// 全局共享状态：ONNX 会话（跨请求复用，需要 &mut 所以包 Mutex）
/// + 特征构建器（build_all 只借用 &self，Arc 共享即可）
#[derive(Clone)]
struct AppState {
    model: Arc<Mutex<Model>>,
    builder: Arc<SnapshotBuilder>,
}

#[derive(Deserialize)]
struct PredictRequest {
    symbol: String,
    /// 预测时点 = win_start_ms + 190_000（T-110s）。调用方负责对齐。
    snap_ms: i64,
}

#[derive(Serialize)]
struct PredictResponse {
    symbol: String,
    snap_ms: i64,
    /// 校准后的 P(UP)，这才是可用于下注的信号
    up_prob: f64,
    down_prob: f64,
    /// ONNX 原始输出（诊断用）
    raw_p: f64,
    /// 特征中 NAN 的数量 —— 过大说明数据缺失严重，调用方可据此过滤
    n_nan: usize,
    /// 调试用：NAN 特征的名字（只在 debug 构建输出）
    nan_features: Vec<String>,
}

/// POST /predict —— 服务端从 ClickHouse 重建快照、算特征、跑模型。
///
/// 不接收客户端传入的特征数组 —— 特征契约在 ONNX 输入名里，
/// 训练和推理共用同一份 compute_features 代码，杜绝 train/serve skew。
async fn predict(
    State(state): State<AppState>,
    Json(req): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, (axum::http::StatusCode, String)> {
    // 校验 symbol 与 snap_ms（防未来信息）
    let snap = state.builder.build_all(&[req.symbol.as_str()], req.snap_ms)
        .map_err(|e| (axum::http::StatusCode::BAD_GATEWAY, format!("快照构建失败: {e}")))?;
    let snap = snap.into_iter().next().expect("build_all 返回空");

    let fv = compute_features(&snap);
    let n_nan = fv.features.iter().filter(|v| v.is_nan()).count();
    let nan_features = fv.features.iter().zip(&fv.feature_names)
        .filter(|(v, _)| v.is_nan())
        .map(|(_, n)| n.clone())
        .collect();

    let mut model = state.model.lock().map_err(|_| {
        (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "模型锁中毒".to_string())
    })?;
    let pred = model.predict(&fv.features)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("推理失败: {e}")))?;

    Ok(Json(PredictResponse {
        symbol: req.symbol,
        snap_ms: req.snap_ms,
        up_prob: pred.calib_p,
        down_prob: 1.0 - pred.calib_p,
        raw_p: pred.raw_p,
        n_nan,
        nan_features,
    }))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env()
            .add_directive("pm_inference=info".parse()?))
        .init();

    let cfg = config::Config::from_env()?;

    // 启动时加载模型 —— 失败直接退出，不对外提供假预测
    let model = Model::load(&cfg.model_path, &cfg.calibration_path)?;
    tracing::info!(
        "模型已加载: {} ({} 个特征)",
        cfg.model_path.display(),
        model.n_features()
    );

    let state = AppState {
        model: Arc::new(Mutex::new(model)),
        builder: Arc::new(SnapshotBuilder::new(format!("http://{}:{}/", cfg.ch_host, cfg.ch_port))),
    };

    let addr = format!("{}:{}", cfg.bind_host, cfg.bind_port);
    let app = Router::new()
        .route("/predict", post(predict))
        .route("/health", axum::routing::get(|| async { "ok" }))
        .with_state(state);

    tracing::info!("inference service listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
