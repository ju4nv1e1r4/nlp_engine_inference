# Inference Engine

The Force is strong with this inference engine! This is a robust, single-threaded Rust binary designed to run AI model inferences using the ONNX Runtime (ORT) directly from the Hugging Face Hub. 

Written purely in Rust, it downloads ONNX models, manages local caching, runs tokenization, and performs inference on the fly—making sure it can execute the Kessel Run in less than 12 parsecs.

## Features

- **Zero-Setup ONNX**: Automatically downloads and bundles the `libonnxruntime` during build. No need to install C++ libraries manually.
- **Hugging Face Hub Integration**: Dynamically downloads only the necessary files (`config.json`, `tokenizer.json`, and `.onnx` files) from any Hugging Face repository.
- **Local Caching**: Models are downloaded once and cached locally in a `./models` directory for blazing-fast subsequent runs.
- **Pure JSON Communication**: Designed for perfect pipeline integrations. Takes JSON as input, outputs the results as JSON to `stdout`, and sends progress/errors securely to `stderr`.
- **Token Classification**: Fully implements BIO (Begin, Inside, Outside) tagging reconstruction for Named Entity Recognition (NER) models.

## How it Works

*Do. Or do not. There is no try.* But if there is an error, this engine handles it gracefully and returns a clean, structured JSON to `stderr` with exit code `1`.

The inference engine follows a strict sequential flow:

1. **Input Parsing**: Reads a JSON input file via the `--input` flag containing the `model` repository ID and the `input` array of strings.
2. **Cache Management**: Checks the local `models/{owner}/{repo}` directory. If the requested files aren't there, it securely downloads them via the HF REST API.
3. **Model Configuration**: Parses `config.json` to automatically infer the target task (e.g., `token-classification`) and loads the `id2label` mappings.
4. **Tokenization**: Loads the local `tokenizer.json` using the Rust `tokenizers` crate to generate `input_ids` and `attention_mask`.
5. **Inference**: Fires up the ORT CPU Session and executes a batch-size-1 inference for each string, returning raw `logits`.
6. **Post-Processing**: The assigned pipeline (like the NER pipeline) parses the `logits`, matches them with the tokenizer's byte offsets, and reconstructs the detected entities.
7. **Output**: Beautifully outputs a structured JSON with the detected labels, offsets, softmax probabilities, and raw ONNX logits.

## Usage

### 1. Build the engine
```bash
cargo build --release
```

### 2. Create an input JSON (`input.json`)
```json
{
  "model": "onnx-community/biomedical-ner-all-ONNX",
  "task": "token-classification",
  "input": [
    "Patient presented with severe headache and fever."
  ]
}
```
*(Note: The `task` field is optional. If omitted, the engine will attempt to auto-infer the task from the model's `config.json`).*

### 3. Run Inference
```bash
./target/release/inference-engine --input input.json
```

If you ever need to clear the local cache and force the engine to redownload the model, you can use the `--refresh-model` flag:
```bash
./target/release/inference-engine --input input.json --refresh-model
```

## Output Structure

The output will only ever be printed to `stdout` to allow easy piping (`> results.json`). All progress bars and warnings are kept strictly in `stderr`.

```json
{
  "model": "onnx-community/biomedical-ner-all-ONNX",
  "task": "token-classification",
  "timestamp": "2026-06-03T22:12:08Z",
  "results": [
    {
      "input": "Patient presented with severe headache and fever.",
      "entities": [
        {
          "text": "severe",
          "label": "Severity",
          "start": 23,
          "end": 29,
          "softmax_prob": 0.9998204,
          "logit_ort": 13.238725
        },
        {
          "text": "headache",
          "label": "Sign_symptom",
          "start": 30,
          "end": 38,
          "softmax_prob": 0.99993575,
          "logit_ort": 13.828054
        }
      ]
    }
  ]
}
```

Enjoy your highly automated, lightweight AI inference tool!
