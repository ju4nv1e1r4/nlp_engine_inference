use crate::error::AppError;
use std::path::Path;

/// Loads tokenizer from `cache_dir/tokenizer.json`.
/// Uses tokenizers::Tokenizer::from_file().
/// Returns AppError::TokenizerError if the file does not exist or is invalid.
pub fn load(cache_dir: &Path) -> Result<tokenizers::Tokenizer, AppError> {
    let path = cache_dir.join("tokenizer.json");
    tokenizers::Tokenizer::from_file(&path)
        .map_err(|e| AppError::TokenizerError(e.to_string()))
}
