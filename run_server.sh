# reference : https://docs.sglang.ai/backend/quantization.html
# reference : https://docs.sglang.ai/backend/server_arguments.html
# reference : https://docs.sglang.ai/backend/hyperparameter_tuning.html

python3 -m sglang.launch_server --model-path Llama-3.2-3B-Instruct-W8A8-Dynamic-Per-Token-V2 \
 --tp 1 \
 --quantization w8a8_int8 \
 --host 0.0.0.0 \
 --context-length 4096 \
 --enable-torch-compile \
 --attention-backend flashinfer \
 --enable-mixed-chunk \
 --schedule-policy fcfs --schedule-conservativeness 0.8




