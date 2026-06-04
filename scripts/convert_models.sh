#!/bin/bash

colors() {
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[0;33m'
  BLUE='\033[0;34m'
  NC='\033[0m'
}
colors

echo -e "${BLUE}[Convert To ONNX] Starting conversion of model to ONNX${NC}"

if [[ ! -d "venv" ]]; then
  python3 -m venv venv
fi

echo -e "${BLUE}[Convert To ONNX] Activating virtual environment${NC}"
source venv/bin/activate

if [[ ! -f "requirements.txt" ]]; then
  pip install optimum[onnxruntime]
  pip freeze > requirements.txt
  echo -e "${GREEN}[Convert To ONNX] Requirements installed${NC}"
fi

read -p "[Convert To ONNX] Enter the model name: " model_name
read -p "[Convert To ONNX] Enter the task: " task

if [[ -z "$model_name" ]]; then
  echo -e "${RED}[Convert To ONNX] Model name cannot be empty${NC}"
  exit 1
fi

if [[ -z "$task" ]]; then
  echo -e "${RED}[Convert To ONNX] Task cannot be empty${NC}"
  exit 1
fi

output_dir="./models/$(echo "$model_name" | tr '/' '_')"
mkdir -p "$output_dir"

echo -e "${BLUE}[Convert To ONNX] Exporting ${model_name} to ${output_dir}${NC}"

optimum-cli export onnx \
  --model "$model_name" \
  --task "$task" \
  "$output_dir"

if [[ $? -eq 0 ]]; then
  echo -e "${GREEN}[Convert To ONNX] Done. Files:${NC}"
  ls -la "$output_dir"
else
  echo -e "${RED}[Convert To ONNX] Export failed${NC}"
  exit 1
fi
