# reference : https://docs.sglang.ai/backend/quantization.html
# reference : https://docs.sglang.ai/backend/server_arguments.html
# reference : https://docs.sglang.ai/backend/hyperparameter_tuning.html

python3 -m sglang.launch_server --model-path llama-3.2-1b-KD-V1 \
 --tp 1 \
 --host 0.0.0.0 \
 --context-length 4096 \
 --enable-torch-compile \
 --attention-backend flashinfer \
 --enable-mixed-chunk \
 --schedule-policy lpm --schedule-conservativeness 0.3
