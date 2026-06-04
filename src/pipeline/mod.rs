pub mod ner;

use crate::error::AppError;
use crate::types::TextResult;
use std::collections::HashMap;

pub trait Pipeline: Send + Sync {
    fn process(
        &self,
        text: &str,
        encoding: &tokenizers::Encoding,
        logits: Vec<Vec<f32>>,
        id2label: &HashMap<String, String>,
    ) -> Result<TextResult, AppError>;
}

/// Returns the correct pipeline for the task.
/// Returns AppError::InvalidInput if the task is not supported.
pub fn for_task(task: &str) -> Result<Box<dyn Pipeline>, AppError> {
    match task {
        "token-classification" => Ok(Box::new(ner::NerPipeline)),
        other => Err(AppError::InvalidInput(format!(
            "Task '{}' not suported in this version.",
            other
        ))),
    }
}
