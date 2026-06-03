use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct InferenceRequest {
    pub model: String,
    pub input: Vec<String>,
    pub task: Option<String>,
}

#[derive(Serialize)]
pub struct InferenceResponse {
    pub model: String,
    pub task: String,
    pub timestamp: String,   // ISO 8601 UTC
    pub results: Vec<TextResult>,
}

#[derive(Serialize)]
pub struct TextResult {
    pub input: String,
    pub entities: Vec<Entity>,
}

#[derive(Serialize)]
pub struct Entity {
    pub text: String,
    pub label: String,
    pub start: usize,
    pub end: usize,
    pub softmax_prob: f32,
    pub logit_ort: f32,
}

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub architectures: Option<Vec<String>>,
    pub id2label: Option<std::collections::HashMap<String, String>>,
}
