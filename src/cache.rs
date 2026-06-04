use crate::error::AppError;
use std::path::PathBuf;

/// Returns the path of the cache directory for a model_id ("owner/repo").
/// Splits on '/' to get owner and repo.
/// Returns AppError::InvalidInput if model_id does not contain '/'.
pub fn cache_dir(model_id: &str) -> Result<PathBuf, AppError> {
    let parts: Vec<&str> = model_id.split('/').collect();
    if parts.len() != 2 {
        return Err(AppError::InvalidInput(format!(
            "Invalid model_id format: {}. Expected 'owner/repo'",
            model_id
        )));
    }

    let current_dir = std::env::current_dir().map_err(|e| AppError::IoError(e.to_string()))?;
    Ok(current_dir.join("models").join(parts[0]).join(parts[1]))
}


/// Removes the complete cache directory for the model_id.
/// Used by the --refresh-model flag.
pub async fn clear(model_id: &str) -> Result<(), AppError> {
    let dir = cache_dir(model_id)?;
    if dir.exists() {
        tokio::fs::remove_dir_all(&dir)
            .await
            .map_err(|e| AppError::IoError(e.to_string()))?;
    }
    Ok(())
}
