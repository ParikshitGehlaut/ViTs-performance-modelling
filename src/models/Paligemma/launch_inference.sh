#!/bin/bash
set -euo pipefail

# -----------------------------
# Paths & settings
# -----------------------------
MODEL_PATH="/export/home/nikhil/parikshit/research/ViTs-performance-modelling/src/models/Paligemma/paligemma-3b-mix-224"
IMAGE_FILE_PATH="/export/home/nikhil/parikshit/research/ViTs-performance-modelling/src/models/Paligemma/test_images/pic2.png"
PROMPT="What is the record national temperature of Iran according to this diagram? "
MAX_TOKENS_TO_GENERATE=1000
TEMPERATURE=0.7
TOP_P=0.9
DO_SAMPLE="False"
ONLY_CPU="False"

# -----------------------------
# Run inference with Python
# -----------------------------
python src/models/Paligemma/inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate "$MAX_TOKENS_TO_GENERATE" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --do_sample "$DO_SAMPLE" \
    --only_cpu "$ONLY_CPU"