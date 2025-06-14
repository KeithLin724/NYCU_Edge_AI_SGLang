#!/bin/bash

# reference : https://docs.sglang.ai/backend/quantization.html
# reference : https://docs.sglang.ai/backend/server_arguments.html
# reference : https://docs.sglang.ai/backend/hyperparameter_tuning.html

MODEL_NAME=${1:-KYLiN724/llama-3.2-1b-KD-V1-W8A8-Dynamic-Per-Token}

python3 -m sglang.launch_server --model-path "$MODEL_NAME" \
 --tp 1 \
 --quantization w8a8_int8 \
 --host 0.0.0.0 \
 --context-length 4096 \
 --enable-torch-compile \
 --attention-backend flashinfer \
 --enable-mixed-chunk \
 --schedule-policy lpm --schedule-conservativeness 0.3
