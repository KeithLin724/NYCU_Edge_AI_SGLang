# NYCU Edge AI Final: SGLang

This project is for the NYCU Edge AI final, focusing on LLM quantization and performance evaluation with SGLang server.

---

## Environment Setup

Make sure you have Conda installed. Create the environment with:

```sh
conda env create -f environment.yml
conda activate edge_ai_sglang
```

---

## How to Run Experiments

### 1. Quantize the Model

Run the following command to quantize the model to INT8:

```sh
python compress_int8.py
```

---

### 2. Throughput Test

1. Start the SG-Lang server:

    ```sh
    sh run_server.sh
    ```

2. Run the throughput test script:

    ```sh
    python result-quant-sglang.py
    ```

---

### 3. Perplexity (PPL) Test

> **Note:** Please shut down the SG-Lang server before running this step.

```sh
python result-quant.py
```

---

## Notes

- Results will be saved to `result_tput.csv` and `result_ppl.csv`.
- You can modify model or dataset parameters at the top of each script.
- If you encounter CUDA out-of-memory errors, try reducing the batch size or sequence length.

---

For any questions, please open an issue or contact the project maintainer.

## Reference

- [SG-Lang](https://docs.sglang.ai/)
- [SG-Lang Github](https://github.com/sgl-project/sglang?tab=readme-ov-file)
- [SGLang 推理引擎技术解析](https://zhuanlan.zhihu.com/p/30886364337)
- [llm-compressor](https://github.com/vllm-project/llm-compressor/tree/main)
