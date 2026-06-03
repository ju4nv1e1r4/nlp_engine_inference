use thiserror::Error;
use serde::Serialize;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("model not found on HF Hub: {0}")]
    ModelNotFound(String),

    #[error("download failed for '{url}': {message}")]
    DownloadFailed { url: String, message: String },

    #[error("task could not be inferred for model '{model_id}'")]
    TaskNotInferred { model_id: String },

    #[error("no ONNX file could be loaded for model '{0}'")]
    OnnxLoadFailed(String),

    #[error("inference failed: {0}")]
    InferenceFailed(String),

    #[error("tokenizer error: {0}")]
    TokenizerError(String),

    #[error("io error: {0}")]
    IoError(String),

    #[error("json error: {0}")]
    JsonError(String),
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<String>,
}

impl From<&AppError> for ErrorResponse {
    fn from(e: &AppError) -> Self {
        match e {
            AppError::TaskNotInferred { model_id } => ErrorResponse {
                error: "task_not_inferred".into(),
                message: format!(
                    "The task could not be automatically inferred for the model. '{}'. \
                     Add a \"task\" field in the JSON input.",
                    model_id
                ),
                suggestion: Some(
                    "Valid values: token-classification, text-classification, \
                     question-answering, fill-mask, text-generation"
                    .into(),
                ),
            },
            AppError::ModelNotFound(id) => ErrorResponse {
                error: "model_not_found".into(),
                message: format!("Model '{}' not found on HF Hub.", id),
                suggestion: None,
            },
            _ => ErrorResponse {
                error: "internal_error".into(),
                message: e.to_string(),
                suggestion: None,
            },
        }
    }
}

impl From<std::io::Error> for AppError {
    fn from(e: std::io::Error) -> Self {
        AppError::IoError(e.to_string())
    }
}

impl From<reqwest::Error> for AppError {
    fn from(e: reqwest::Error) -> Self {
        AppError::DownloadFailed {
            url: e.url().map(|u| u.to_string()).unwrap_or_else(|| "unknown".into()),
            message: e.to_string(),
        }
    }
}

impl From<serde_json::Error> for AppError {
    fn from(e: serde_json::Error) -> Self {
        AppError::JsonError(e.to_string())
    }
}
