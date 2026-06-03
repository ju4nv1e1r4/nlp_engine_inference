mod cache;
mod error;
mod hub;
mod model;
mod pipeline;
mod tokenizer;
mod types;

use clap::Parser;
use crate::error::{AppError, ErrorResponse};
use crate::types::{InferenceRequest, InferenceResponse, ModelConfig};
use reqwest::Client;
use std::path::PathBuf;
use std::collections::HashMap;

#[derive(Parser)]
#[command(name = "inference-engine", about = "ONNX inference engine via HF Hub")]
struct Cli {
    #[arg(long)]
    input: PathBuf,

    #[arg(long, default_value_t = false)]
    refresh_model: bool,
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    if let Err(e) = run().await {
        let response = ErrorResponse::from(&e);
        let json = serde_json::to_string(&response).unwrap_or_else(|_| {
            r#"{"error":"serialization_failed","message":"failed to serialize error"}"#.into()
        });
        eprintln!("{}", json);
        std::process::exit(1);
    }
}

async fn run() -> Result<(), AppError> {
    let cli = Cli::parse();

    let input_content = tokio::fs::read_to_string(&cli.input)
        .await
        .map_err(|e| AppError::IoError(format!("failed to read input: {}", e)))?;

    let request: InferenceRequest = serde_json::from_str(&input_content)
        .map_err(|e| AppError::JsonError(format!("invalid input JSON: {}", e)))?;

    if !request.model.contains('/') {
        return Err(AppError::InvalidInput("model id must contain '/'".into()));
    }
    if request.input.is_empty() {
        return Err(AppError::InvalidInput("input array must not be empty".into()));
    }

    let client = Client::new();

    if cli.refresh_model {
        cache::clear(&request.model).await?;
    }

    let cache_dir = cache::cache_dir(&request.model)?;

    eprintln!("[inference-engine] Downloading files for model {}...", request.model);
    eprintln!("[inference-engine] Warning: ONNX models may be large. Wait for download to complete.");

    let siblings = hub::list_model_files(&client, &request.model).await?;

    let mandatory_files = ["config.json", "tokenizer.json"];
    let optional_files = ["tokenizer_config.json", "special_tokens_map.json"];

    for file in mandatory_files {
        let dest = cache_dir.join(file);
        if !dest.exists() {
            hub::download_file(&client, &request.model, file, &dest).await?;
        }
    }

    for file in optional_files {
        let dest = cache_dir.join(file);
        if !dest.exists() {
            if siblings.iter().any(|s| s.rfilename == file) {
                hub::download_file(&client, &request.model, file, &dest).await?;
            }
        }
    }

    let config_content = tokio::fs::read_to_string(cache_dir.join("config.json"))
        .await
        .map_err(|e| AppError::IoError(e.to_string()))?;

    let config: ModelConfig = serde_json::from_str(&config_content)
        .map_err(|e| AppError::JsonError(format!("config.json inválido: {}", e)))?;

    let task = infer_task(&config, &request.task, &request.model)?;

    let tokenizer = tokenizer::load(&cache_dir)?;

    let mut onnx_files: Vec<&hub::HfFile> = siblings
        .iter()
        .filter(|f| f.rfilename.ends_with(".onnx"))
        .collect();

    onnx_files.sort_by_key(|f| f.size.unwrap_or(u64::MAX));

    if onnx_files.is_empty() {
        return Err(AppError::OnnxLoadFailed(request.model.clone()));
    }

    let mut session: Option<ort::session::Session> = None;
    for onnx_file in &onnx_files {
        let onnx_path = cache_dir.join(&onnx_file.rfilename);
        if !onnx_path.exists() {
            hub::download_file(&client, &request.model, &onnx_file.rfilename, &onnx_path).await?;
        }

        eprintln!("[inference-engine] Download is done. Loading model...");
        match model::load_session(&onnx_path) {
            Ok(s) => {
                session = Some(s);
                break;
            }
            Err(e) => {
                eprintln!("[inference-engine] Failed to load {}: {}. Trying next file...", onnx_file.rfilename, e);
            }
        }
    }

    let mut session = session.ok_or_else(|| AppError::OnnxLoadFailed(request.model.clone()))?;

    let id2label = config.id2label.unwrap_or_else(HashMap::new);
    let pipeline = pipeline::for_task(&task)?;

    let mut results = Vec::with_capacity(request.input.len());

    for text in &request.input {
        let encoding = tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| AppError::TokenizerError(e.to_string()))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();

        let logits = model::run(&mut session, &input_ids, &attention_mask)?;

        let text_result = pipeline.process(text, &encoding, logits, &id2label)?;
        results.push(text_result);
    }

    let response = InferenceResponse {
        model: request.model,
        task,
        timestamp: chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        results,
    };

    let output_json = serde_json::to_string_pretty(&response)
        .map_err(|e| AppError::JsonError(e.to_string()))?;

    println!("{}", output_json);

    Ok(())
}

fn infer_task(config: &ModelConfig, request_task: &Option<String>, model_id: &str) -> Result<String, AppError> {
    if let Some(t) = request_task {
        return Ok(t.clone());
    }
    let arch_map = [
        ("ForTokenClassification", "token-classification"),
        ("ForSequenceClassification", "text-classification"),
        ("ForQuestionAnswering", "question-answering"),
        ("ForMaskedLM", "fill-mask"),
        ("ForCausalLM", "text-generation"),
    ];
    if let Some(archs) = &config.architectures {
        for arch in archs {
            for (suffix, task) in &arch_map {
                if arch.ends_with(suffix) {
                    return Ok(task.to_string());
                }
            }
        }
    }
    if let Some(id2label) = &config.id2label {
        let has_bio = id2label.values().any(|v| v.starts_with("B-") || v.starts_with("I-"));
        if has_bio {
            return Ok("token-classification".to_string());
        }
    }
    Err(AppError::TaskNotInferred { model_id: model_id.to_string() })
}
