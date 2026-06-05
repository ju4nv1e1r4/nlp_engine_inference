use crate::error::AppError;
use crate::pipeline::Pipeline;
use crate::types::{Entity, TextResult};
use std::collections::HashMap;

pub struct NerPipeline;

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|x| x / sum).collect()
}

struct EntityAccumulator {
    label_type: String,
    char_start: usize,
    char_end: usize,
    logits: Vec<f32>,
    probs: Vec<f32>,
}

impl Pipeline for NerPipeline {
    fn process(
        &self,
        text: &str,
        encoding: &tokenizers::Encoding,
        logits: Vec<Vec<f32>>,
        id2label: &HashMap<String, String>,
    ) -> Result<TextResult, AppError> {
        let mut current: Option<EntityAccumulator> = None;
        let mut entities = Vec::new();

        let word_ids = encoding.get_word_ids();
        let offsets = encoding.get_offsets();

        for i in 0..logits.len() {
            if word_ids[i].is_none() {
                continue;
            }

            let token_logits = &logits[i];
            let mut pred_idx = 0;
            let mut max_val = f32::NEG_INFINITY;
            for (idx, &val) in token_logits.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    pred_idx = idx;
                }
            }

            let logit_ort = max_val;
            let softmax_probs = softmax(token_logits);
            let softmax_prob = softmax_probs[pred_idx];

            let label = id2label
                .get(&pred_idx.to_string())
                .cloned()
                .unwrap_or_else(|| "O".to_string());

            let (char_start, char_end) = offsets[i];

            if label == "O" {
                if let Some(acc) = current.take() {
                    entities.push(Entity {
                        text: text[acc.char_start..acc.char_end].to_string(),
                        label: acc.label_type,
                        start: acc.char_start,
                        end: acc.char_end,
                        softmax_prob: acc.probs.iter().sum::<f32>() / acc.probs.len() as f32,
                        logit_ort: acc.logits.iter().sum::<f32>() / acc.logits.len() as f32,
                    });
                }
            } else if label.starts_with("B-") {
                let label_type = &label[2..];
                let mut should_start_new = false;

                if let Some(mut acc) = current.take() {
                    if acc.label_type == label_type && acc.char_end == char_start {
                        acc.char_end = char_end;
                        acc.logits.push(logit_ort);
                        acc.probs.push(softmax_prob);
                        current = Some(acc);
                    } else {
                        entities.push(Entity {
                            text: text[acc.char_start..acc.char_end].to_string(),
                            label: acc.label_type,
                            start: acc.char_start,
                            end: acc.char_end,
                            softmax_prob: acc.probs.iter().sum::<f32>() / acc.probs.len() as f32,
                            logit_ort: acc.logits.iter().sum::<f32>() / acc.logits.len() as f32,
                        });
                        should_start_new = true;
                    }
                } else {
                    should_start_new = true;
                }

                if should_start_new {
                    current = Some(EntityAccumulator {
                        label_type: label_type.to_string(),
                        char_start,
                        char_end,
                        logits: vec![logit_ort],
                        probs: vec![softmax_prob],
                    });
                }
            } else if label.starts_with("I-") {
                let label_type = &label[2..];
                let mut should_start_new = false;

                if let Some(mut acc) = current.take() {
                    if acc.label_type == label_type {
                        acc.char_end = char_end;
                        acc.logits.push(logit_ort);
                        acc.probs.push(softmax_prob);
                        current = Some(acc);
                    } else {
                        entities.push(Entity {
                            text: text[acc.char_start..acc.char_end].to_string(),
                            label: acc.label_type,
                            start: acc.char_start,
                            end: acc.char_end,
                            softmax_prob: acc.probs.iter().sum::<f32>() / acc.probs.len() as f32,
                            logit_ort: acc.logits.iter().sum::<f32>() / acc.logits.len() as f32,
                        });
                        should_start_new = true;
                    }
                } else {
                    should_start_new = true;
                }

                if should_start_new {
                    current = Some(EntityAccumulator {
                        label_type: label_type.to_string(),
                        char_start,
                        char_end,
                        logits: vec![logit_ort],
                        probs: vec![softmax_prob],
                    });
                }
            }
        }

        if let Some(acc) = current.take() {
            entities.push(Entity {
                text: text[acc.char_start..acc.char_end].to_string(),
                label: acc.label_type,
                start: acc.char_start,
                end: acc.char_end,
                softmax_prob: acc.probs.iter().sum::<f32>() / acc.probs.len() as f32,
                logit_ort: acc.logits.iter().sum::<f32>() / acc.logits.len() as f32,
            });
        }

        Ok(TextResult {
            input: text.to_string(),
            entities,
        })
    }
}
