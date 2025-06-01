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

### 1. Get the Model

You have three options to obtain and prepare the model:

#### 1. Download the pre-quantized (W8A8 INT8) model

- **Model link:** [KYLiN724/llama-3.2-1b-KD-V1-W8A8-Dynamic-Per-Token](https://huggingface.co/KYLiN724/llama-3.2-1b-KD-V1-W8A8-Dynamic-Per-Token)
- **How to build it yourself:** See [BUILD_MODEL.md](./BUILD_MODEL.md)
- **Download with script:**

    ```sh
    python get_1b_model_int8.py
    ```

---

#### 2. Download the non-quantized (FP16) model

- **Model link:** [hlhsiao/llama-3.2-1b-KD-V1](https://huggingface.co/hlhsiao/llama-3.2-1b-KD-V1)
- **How to build it yourself:** See [build_small_model/README.md](./build_small_model/README.md)
- **Download with script:**

    ```sh
    python get_1b_model.py
    ```

---

#### 3. Quantize the non-quantized model to INT8

If you have downloaded the non-quantized model, you can quantize it to INT8 by running:

```sh
python compress_int8.py
```

The quantized model will be saved to a new directory for further use or uploading to Hugging Face.

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
