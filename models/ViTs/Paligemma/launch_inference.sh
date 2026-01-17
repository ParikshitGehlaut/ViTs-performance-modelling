#!/bin/bash

MODEL_PATH="/home/parikshit/ViTs-performance-modelling/models/ViTs/Paligemma/paligemma-3b-mix-224/"
PROMPT="The name of the tower is "
IMAGE_FILE_PATH="/home/parikshit/ViTs-performance-modelling/models/ViTs/Paligemma/test_images/pic1.jpg"
MAX_TOKENS_TO_GENERATE=1000
TEMPERATURE=0.7
TOP_P=0.9
DO_SAMPLE="False"
ONLY_CPU="False"

python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU \