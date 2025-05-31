python3 -m sglang.launch_server --model-path Llama-3.2-3B-Instruct-W8A8-Dynamic-Per-Token-One \
 --tp 2 \
 --host 0.0.0.0 \
 --context-length 4096 \
 --enable-torch-compile \
 --attention-backend flashinfer \
 --enable-mixed-chunk 