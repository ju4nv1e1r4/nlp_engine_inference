use crate::error::AppError;
use ort::session::Session;
use std::path::Path;

/// Loads a ORT session from a .onnx file.
/// Configures CPU execution (no special execution providers).
pub fn load_session(onnx_path: &Path) -> Result<Session, AppError> {
    Session::builder()
        .map_err(|e: ort::Error| AppError::OnnxLoadFailed(e.to_string()))?
        .commit_from_file(onnx_path)
        .map_err(|e: ort::Error| AppError::OnnxLoadFailed(e.to_string()))
}

/// Executes a inference with batch_size=1.
///
/// Params:
///   - session: ORT session loaded
///   - input_ids: Vec<i64> with token IDs
///   - attention_mask: Vec<i64> with attention mask
///
/// Returns: Vec<Vec<f32>> with shape [seq_len][num_labels] (logits per token)
pub fn run(
    session: &mut Session,
    input_ids: &[i64],
    attention_mask: &[i64],
    token_type_ids: &[i64],
) -> Result<Vec<Vec<f32>>, AppError> {
    let seq_len = input_ids.len();

    let ids_tensor = ort::value::Tensor::from_array(([1, seq_len], input_ids.to_vec()))
        .map_err(|e: ort::Error| AppError::InferenceFailed(e.to_string()))?;
    let mask_tensor = ort::value::Tensor::from_array(([1, seq_len], attention_mask.to_vec()))
        .map_err(|e: ort::Error| AppError::InferenceFailed(e.to_string()))?;

    let expects_type_ids = session.inputs().iter().any(|i| i.name() == "token_type_ids");

    let outputs = if expects_type_ids {
        let type_tensor = ort::value::Tensor::from_array(([1, seq_len], token_type_ids.to_vec()))
            .map_err(|e: ort::Error| AppError::InferenceFailed(e.to_string()))?;
        session.run(ort::inputs![
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor,
            "token_type_ids" => type_tensor,
        ]).map_err(|e: ort::Error| AppError::InferenceFailed(e.to_string()))?
    } else {
        session.run(ort::inputs![
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor,
        ]).map_err(|e: ort::Error| AppError::InferenceFailed(e.to_string()))?
    };

    let logits_val = outputs
        .get("logits")
        .ok_or_else(|| AppError::InferenceFailed("Tensor 'logits' not found in outputs".into()))?;

    let (_, data) = logits_val
        .try_extract_tensor::<f32>()
        .map_err(|e: ort::Error| AppError::InferenceFailed(format!("Failed to extract logits tensor: {}", e)))?;

    let total_elements = data.len();
    if total_elements % seq_len != 0 {
        return Err(AppError::InferenceFailed("Unexpected output tensor size".into()));
    }
    let num_labels = total_elements / seq_len;

    let mut result = Vec::with_capacity(seq_len);
    for i in 0..seq_len {
        let mut token_logits = Vec::with_capacity(num_labels);
        for j in 0..num_labels {
            token_logits.push(data[i * num_labels + j]);
        }
        result.push(token_logits);
    }

    Ok(result)
}
