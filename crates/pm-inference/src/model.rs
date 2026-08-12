//! ONNX 模型加载 + 推理 + isotonic 校准。
//!
//! 推理端只依赖两个文件：
//!   - `model.onnx`          —— research/train_*.py 导出的模型（特征契约）
//!   - `calibration.json`    —— 配套的 isotonic 校准器
//!
//! 模型无关性
//! ----------
//! 支持两种 ONNX 输出签名，加载时按类型自动判定：
//!   - `sequence<map<int64,float>>` —— onnxmltools 树模型（LightGBM）的
//!     ZipMap 输出，每行 `{0: P(down), 1: P(up)}`
//!   - `tensor<float>[N, 2]` / `[N, 1]` —— PyTorch 等导出的普通张量
//!     （TCN-LSTM 走这条路）
//! 换模型时只要 ONNX 输出属于以上任一种，推理端不用改。
//!
//! 校准语义必须与训练端完全一致：sklearn 的 IsotonicRegression
//! (out_of_bounds='clip') 等价于对 (x, y) 阈值表做线性插值，
//! 超过范围时钳制到端点 —— 与 numpy.interp 一致。

use std::path::Path;

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::{MapValueType, Tensor, TensorElementType, ValueType};
use serde::Deserialize;

/// ONNX 输出的提取方式 —— 加载时按输出类型判定一次，推理时不再猜。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputKind {
    /// `sequence<map<int64,float>>`：树模型的 ZipMap，取 key=1
    ZipMap,
    /// `tensor<float>[N, 2]`：二分类概率张量，取第 1 列
    Tensor2,
    /// `tensor<float>[N, 1]`：单列概率张量，直接取
    Tensor1,
}

/// isotonic 校准器（对应 train.py 写出的 calibration.json）
#[derive(Debug, Deserialize)]
pub struct Calibration {
    pub method: String,
    /// 原始模型输出 P_raw（训练端 val 集上的分位点）
    pub x: Vec<f64>,
    /// 校准后概率（与 x 一一对应）
    pub y: Vec<f64>,
    pub raw_min: f64,
    pub raw_max: f64,
    pub n: usize,
}

/// 加载好的模型 + 校准器，跨请求共享（axum state 里是 Clone）
pub struct Model {
    session: Session,
    calib: Calibration,
    n_features: usize,
    input_name: String,
    output_name: String,
    output_kind: OutputKind,
}

/// 一次预测的完整输出
#[derive(Debug)]
pub struct Prediction {
    /// ONNX 原始输出 P(UP)，未经校准
    pub raw_p: f64,
    /// 校准后概率（0..1，可解释的 P(UP)）
    pub calib_p: f64,
}

impl Model {
    /// 从磁盘加载 ONNX 模型和校准器。
    ///
    /// 启动时调用一次，加载失败直接报错退出 —— 模型文件缺失或损坏时
    /// 不应继续对外提供 0.5 的假预测。
    pub fn load(model_path: &Path, calib_path: &Path) -> Result<Self> {
        let session = Session::builder()
            .with_context(|| "初始化 ONNX runtime 失败")?
            .commit_from_file(model_path)
            .with_context(|| format!("加载 ONNX 模型失败: {}", model_path.display()))?;

        // 从 ONNX 图里读输入名 —— 特征契约
        let input_name = session.inputs()
            .first()
            .map(|o| o.name().to_owned())
            .context("ONNX 模型没有输入节点")?;

        // 输出按「类型」而非名字挑选 —— 不同训练框架的导出器命名不一样
        // （树模型叫 probabilities，PyTorch 可能叫 output/logits）。
        // 名字只作为同类型多输出时的优先级提示。
        let (output_name, output_kind) = Self::pick_output(&session)?;

        // 输入特征数（形状 [None, N] 的第二维）。ort 2.0 的 Outlet 不暴露
        // shape，只有 dtype → ValueType::tensor_shape()；动态维度为 -1。
        let in_type = session.inputs()[0].dtype();
        let n_features = in_type.tensor_shape()
            .context("ONNX 输入不是张量")?
            .get(1)
            .copied()
            .context("ONNX 输入形状没有特征维度")?
            .try_into()
            .context("ONNX 特征维度不是 usize")?;

        let calib: Calibration = serde_json::from_str(
            &std::fs::read_to_string(calib_path)
                .with_context(|| format!("读取校准器失败: {}", calib_path.display()))?
        ).context("calibration.json 格式错误")?;

        // 校准表至少 1 个点（n==1 是退化情况：模型输出为常数，映射到单点）
        anyhow::ensure!(
            calib.x.len() >= 1 && calib.x.len() == calib.y.len(),
            "calibration.json 阈值表点数不足或不匹配"
        );
        // x 必须单调不减（isotonic 的输入；允许重复阈值）
        anyhow::ensure!(
            calib.x.windows(2).all(|w| w[0] <= w[1]),
            "calibration.json 的 x 非单调"
        );

        Ok(Self { session, calib, n_features, input_name, output_name, output_kind })
    }

    /// 在 ONNX 的输出里挑出概率输出，并判定提取方式。
    ///
    /// 判定只看类型，不看名字 —— 换模型时导出器命名会变，类型不会：
    ///   - `sequence<map<..>>`   → ZipMap（树模型）
    ///   - `tensor<float>[_, 2]` → Tensor2（二分类概率）
    ///   - `tensor<float>[_, 1]` → Tensor1（单列概率）
    /// 整数张量（label 输出）直接排除。同类型有多个候选时，名字里带
    /// prob 的优先，否则取第一个。
    fn pick_output(session: &Session) -> Result<(String, OutputKind)> {
        let mut candidates: Vec<(String, OutputKind)> = Vec::new();

        for out in session.outputs().iter() {
            let kind = match out.dtype() {
                ValueType::Sequence(inner) if matches!(**inner, ValueType::Map { .. }) => {
                    Some(OutputKind::ZipMap)
                }
                ValueType::Tensor { ty, shape, .. }
                    if matches!(ty, TensorElementType::Float32 | TensorElementType::Float64)
                        && matches!(shape.last(), Some(2)) =>
                {
                    Some(OutputKind::Tensor2)
                }
                ValueType::Tensor { ty, shape, .. }
                    if matches!(ty, TensorElementType::Float32 | TensorElementType::Float64)
                        && matches!(shape.last(), Some(1)) =>
                {
                    Some(OutputKind::Tensor1)
                }
                _ => None,   // label、非浮点张量等一律排除
            };
            if let Some(k) = kind {
                candidates.push((out.name().to_owned(), k));
            }
        }

        anyhow::ensure!(
            !candidates.is_empty(),
            "ONNX 模型没有可用作概率的输出（期望 sequence<map> 或 float 张量 [_,2]/[_,1]）"
        );

        // 名字含 prob 的优先 —— 多输出模型里更可能是我们要的那个
        let idx = candidates.iter()
            .position(|(n, _)| n.to_ascii_lowercase().contains("prob"))
            .unwrap_or(0);
        Ok(candidates.swap_remove(idx))
    }

    pub fn output_kind(&self) -> OutputKind { self.output_kind }

    pub fn n_features(&self) -> usize { self.n_features }

    /// 运行推理：33 维特征向量 → 校准后 P(UP)。
    pub fn predict(&mut self, features: &[f64]) -> Result<Prediction> {
        anyhow::ensure!(
            features.len() == self.n_features,
            "特征维度不符：输入 {}，模型期望 {}",
            features.len(), self.n_features
        );

        // 特征里可能有 NAN（缺失），ONNX 不接受 NAN —— 用 0 填充。
        // LightGBM 训练时把缺失特征当独立分支处理，这里的 0 只是
        // 占位；真实语义由训练侧的缺失处理保证。
        let feats_f32: Vec<f32> = features.iter()
            .map(|v| if v.is_nan() { 0.0f32 } else { *v as f32 })
            .collect();

        // [1, N] 张量
        let input = Tensor::from_array(([1usize, self.n_features], feats_f32))
            .context("构造输入张量失败")?;

        let outputs = self.session.run(ort::inputs![self.input_name.as_str() => input])
            .with_context(|| format!("ONNX 推理失败（输入名: {}）", self.input_name))?;

        // 按加载时判定的类型提取 P(UP) —— 树模型走 ZipMap，
        // PyTorch 等走普通张量，两条路都不依赖输出节点的名字。
        let prob_out = outputs
            .get(&self.output_name)
            .with_context(|| format!("输出缺少 {}", self.output_name))?;

        let raw_p = match self.output_kind {
            OutputKind::ZipMap => {
                // onnxmltools 的 binary 输出是 sequence<map<int64, float>>：
                // 每个样本一个 dict {0: P(down), 1: P(up)}
                let seq = prob_out.try_extract_sequence::<MapValueType<i64, f32>>()
                    .with_context(|| "输出不是 sequence<map<int64, float>>")?;
                anyhow::ensure!(seq.len() == 1, "概率序列长度异常: {}", seq.len());
                let map = seq[0].extract_map();   // 类型由 MapValueType<i64, f32> 推断
                *map.get(&1).context("概率 map 缺少 key 1 (P(up))")? as f64
            }
            OutputKind::Tensor2 => {
                // [1, 2] = [P(down), P(up)]
                let (_, data) = prob_out.try_extract_tensor::<f32>()
                    .with_context(|| "输出不是 float 张量")?;
                anyhow::ensure!(data.len() == 2, "概率张量长度异常: {}（期望 2）", data.len());
                data[1] as f64
            }
            OutputKind::Tensor1 => {
                // [1, 1] = [P(up)]
                let (_, data) = prob_out.try_extract_tensor::<f32>()
                    .with_context(|| "输出不是 float 张量")?;
                anyhow::ensure!(data.len() == 1, "概率张量长度异常: {}（期望 1）", data.len());
                data[0] as f64
            }
        };

        // 概率必须落在 [0,1] —— 越界说明模型导出时漏了 sigmoid/softmax，
        // 直接报错好过把 logits 当概率喂给校准器。
        anyhow::ensure!(
            (0.0..=1.0).contains(&raw_p),
            "模型输出 {raw_p} 不在 [0,1]，可能导出时漏了 sigmoid/softmax"
        );

        Ok(Prediction {
            raw_p,
            calib_p: isotonic_clip(raw_p, &self.calib),
        })
    }
}

/// 复现 sklearn IsotonicRegression(out_of_bounds='clip') + np.interp 的语义：
/// 在 (x, y) 分段线性插值，超出 x 范围时钳制到两端。
/// 单点表（退化模型输出为常数）等价于常数映射。
pub fn isotonic_clip(raw_p: f64, calib: &Calibration) -> f64 {
    let x = &calib.x;
    let y = &calib.y;
    let n = x.len();

    if n == 1 { return y[0]; }   // 退化情况：常数映射

    if raw_p <= x[0] { return y[0]; }
    if raw_p >= x[n - 1] { return y[n - 1]; }

    // 二分查找 raw_p 所在区间
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if raw_p >= x[mid] { lo = mid; } else { hi = mid; }
    }
    let x0 = x[lo]; let x1 = x[hi];
    let y0 = y[lo]; let y1 = y[hi];
    // 线性插值（x1 > x0 已由 load() 保证）
    y0 + (y1 - y0) * (raw_p - x0) / (x1 - x0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calib() -> Calibration {
        // 手工构造：3 个点，模拟训练端 isotonic 输出
        Calibration {
            method: "isotonic_clip".into(),
            x: vec![0.2, 0.5, 0.8],
            y: vec![0.1, 0.5, 0.9],
            raw_min: 0.2,
            raw_max: 0.8,
            n: 3,
        }
    }

    #[test]
    fn interpolation_in_between() {
        let c = calib();
        // (0.2,0.1)-(0.5,0.5)：中点 0.35 → 0.3
        assert!((isotonic_clip(0.35, &c) - 0.3).abs() < 1e-12);
        // (0.5,0.5)-(0.8,0.9)：0.65 → 0.7
        assert!((isotonic_clip(0.65, &c) - 0.7).abs() < 1e-12);
    }

    #[test]
    fn clamps_out_of_range() {
        let c = calib();
        assert_eq!(isotonic_clip(0.0, &c), 0.1, "低于下界钳到 y[0]");
        assert_eq!(isotonic_clip(0.1, &c), 0.1, "低于下界钳到 y[0]");
        assert_eq!(isotonic_clip(0.9, &c), 0.9, "高于上界钳到 y[last]");
        assert_eq!(isotonic_clip(1.0, &c), 0.9, "高于上界钳到 y[last]");
    }

    #[test]
    fn exact_thresholds() {
        let c = calib();
        assert_eq!(isotonic_clip(0.2, &c), 0.1);
        assert_eq!(isotonic_clip(0.5, &c), 0.5);
        assert_eq!(isotonic_clip(0.8, &c), 0.9);
    }

    #[test]
    fn single_point_calibration_is_constant() {
        // 退化情况：val 上模型输出为常数（如数据量太少时），
        // isotonic 只剩一个点 → 校准等价于常数映射。
        let c = Calibration {
            method: "isotonic_clip".into(),
            x: vec![0.48],
            y: vec![0.581],
            raw_min: 0.48,
            raw_max: 0.48,
            n: 31,
        };
        assert_eq!(isotonic_clip(0.0, &c), 0.581);
        assert_eq!(isotonic_clip(0.48, &c), 0.581);
        assert_eq!(isotonic_clip(1.0, &c), 0.581);
    }

    #[test]
    fn duplicate_thresholds_are_allowed() {
        // isotonic 可能产生重复阈值（同 x 不同 y 被合并）—— load() 应接受。
        // 插值对 x0 == x1 的区间退化（这里直接用点查验证单点语义）。
        let c = Calibration {
            method: "isotonic_clip".into(),
            x: vec![0.3, 0.5, 0.5, 0.7],
            y: vec![0.2, 0.4, 0.4, 0.6],
            raw_min: 0.3,
            raw_max: 0.7,
            n: 4,
        };
        // 0.5 落在重复点上 → 应返回 0.4
        assert!((isotonic_clip(0.5, &c) - 0.4).abs() < 1e-12);
        // 0.4 在 (0.3,0.2)-(0.5,0.4) 之间 → 0.3
        assert!((isotonic_clip(0.4, &c) - 0.3).abs() < 1e-12);
    }

    #[test]
    fn zipmap_model_kind_is_detected() {
        // verify_onnx 在 Python 端确认 LightGBM 导出是 sequence<map>，
        // 这里用 synth 模型验证 Rust 端对 ZipMap 的自动判定。
        if !Path::new("/tmp/synth_model.onnx").exists() {
            eprintln!("跳过：缺少 /tmp/synth_model.onnx（先运行 research/synth/gen_synth_model.py）");
            return;
        }
        let session = Session::builder().unwrap()
            .commit_from_file("/tmp/synth_model.onnx").unwrap();
        let (name, kind) = Model::pick_output(&session).unwrap();
        assert_eq!(kind, OutputKind::ZipMap, "LightGBM 输出应为 ZipMap，实际 {kind:?} ({name})");
        assert_eq!(name, "probabilities");
    }

    #[test]
    fn onnx_end_to_end_matches_python() {
        // 冒烟测试：加载真实 ONNX + calibration（合成模型），与 Python
        // onnxruntime 的参考输出对比（diff 1e-6）。模型文件由
        // research/synth/gen_synth_model.py 生成。
        let model_path = Path::new("/tmp/synth_model.onnx");
        let calib_path = Path::new("/tmp/calibration.json");
        if !model_path.exists() {
            eprintln!("跳过：缺少 /tmp/synth_model.onnx（先运行 research/synth/gen_synth_model.py）");
            return;
        }
        let mut model = Model::load(model_path, calib_path).unwrap();

        // 5 行参考输入（与 Python 端 /tmp/synth_X5.raw 一致：f32 LE 字节）
        let x5: Vec<u8> = std::fs::read("/tmp/synth_X5.raw").unwrap();
        assert_eq!(x5.len(), 5 * 33 * 4);
        let mut feats = vec![0f64; 5 * 33];
        for (i, chunk) in x5.chunks_exact(4).enumerate() {
            feats[i] = f32::from_le_bytes(chunk.try_into().unwrap()) as f64;
        }
        // Python 参考输出（onnxruntime 对同一 X5 的 P(up)，f64 LE 字节）
        let ref_bytes: Vec<u8> = std::fs::read("/tmp/synth_pup5.raw").unwrap();
        let ref_pup: Vec<f64> = ref_bytes.chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(ref_pup.len(), 5);

        for (row, expected) in feats.chunks_exact(33).zip(ref_pup.iter()) {
            let p = model.predict(row).unwrap();
            let diff = (p.raw_p - expected).abs();
            assert!(diff < 1e-6,
                "raw_p 与 Python 不一致: rust={:.9} python={:.9} diff={:.2e}",
                p.raw_p, expected, diff);
        }
    }
}
