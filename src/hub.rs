use crate::error::AppError;
use serde::Deserialize;

const HF_BASE_URL: &str = "https://huggingface.co";

#[derive(Deserialize)]
struct HfModelInfo {
    siblings: Vec<HfFile>,
}

#[derive(Deserialize, Clone)]
pub struct HfFile {
    pub rfilename: String,
    pub size: Option<u64>,
}

/// Returns a list of all files in the HF repository.
/// Returns AppError::ModelNotFound if the API returns 404.
pub async fn list_model_files(
    client: &reqwest::Client,
    model_id: &str,
) -> Result<Vec<HfFile>, AppError> {
    let url = format!("{}/api/models/{}", HF_BASE_URL, model_id);
    let response = client.get(&url).send().await?;

    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(AppError::ModelNotFound(model_id.to_string()));
    }

    if !response.status().is_success() {
        return Err(AppError::DownloadFailed {
            url,
            message: format!("HTTP error: {}", response.status()),
        });
    }

    let info: HfModelInfo = response.json().await?;
    Ok(info.siblings)
}

/// Download a file from the HF Hub and saves it to `dest`.
/// Uses response.bytes().await to download the file before saving.
/// Displays a progress message on stderr before starting.
pub async fn download_file(
    client: &reqwest::Client,
    model_id: &str,
    filename: &str,
    dest: &std::path::Path,
) -> Result<(), AppError> {
    eprintln!("[inference-engine] Downloading {}...", filename);

    let url = format!("{}/{}/resolve/main/{}", HF_BASE_URL, model_id, filename);
    let response = client.get(&url).send().await?;

    if !response.status().is_success() {
        return Err(AppError::DownloadFailed {
            url,
            message: format!("HTTP error: {}", response.status()),
        });
    }

    let bytes = response.bytes().await?;

    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    tokio::fs::write(dest, bytes).await?;

    Ok(())
}
