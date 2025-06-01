# NYCU Edge AI Final: SGLang

This project is for the NYCU Edge AI final, focusing on LLM quantization and performance evaluation with SGLang server.

![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black) ![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)![nVIDIA](https://img.shields.io/badge/cuda-000000.svg?style=for-the-badge&logo=nVIDIA&logoColor=green)

---

## Environment Setup

Make sure you have Conda installed. Create the environment with:

```sh
conda env create -f environment.yml
conda activate edge_ai_sglang_stable
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
    python get_preprocess_model.py --quant
    ```

---

#### 2. Download the non-quantized (FP16) model

- **Model link:** [hlhsiao/llama-3.2-1b-KD-V1](https://huggingface.co/hlhsiao/llama-3.2-1b-KD-V1)
- **How to build it yourself:** See [build_small_model/README.md](./build_small_model/README.md)
- **Download with script:**

    ```sh
    python get_preprocess_model.py
    ```

---

#### 3. Quantize the non-quantized model to INT8

If you have downloaded the non-quantized model, you can quantize it to INT8 by running:

```sh
# Quantize the default model (meta-llama/Llama-3.2-3B-Instruct)
python compress_to_int8.py

# Or, quantize a custom model by specifying the model name or path
# python compress_to_int8.py --model_name <model_name_or_path>
```

The quantized model will be saved to a new directory for further use or uploading to Hugging Face.

---

### 2. Throughput Test

1. Start the SG-Lang server:

    ```sh
    # Start the SG-Lang server with the default pre-built model (auto-download if not present)
    sh run_server.sh

    # Or, specify a custom model path or Hugging Face repo
    # sh run_server.sh <model_name_or_path>
    ```

2. Run the throughput test script:

    ```sh
    # Run throughput test with the default pre-built model
    python result-quant-sglang.py

    # Or, specify a custom model path or Hugging Face repo
    # python result-quant-sglang.py --model_name <model_name_or_path>
    ```

---

### 3. Perplexity (PPL) Test

> **Note:** Please shut down the SG-Lang server before running this step.

```sh
# Run perplexity (PPL) test with the default pre-built model
python result-quant.py

# Or, specify a custom model path or Hugging Face repo
# python result-quant.py --model_name <model_name_or_path>
```

---

## Notes

- Results will be saved to `result_tput.csv` and `result_ppl.csv`.
- You can modify model or dataset parameters at the top of each script.
- If you encounter CUDA out-of-memory errors, try reducing the batch size or sequence length.

---

For any questions, please open an issue or contact the project maintainer.

## Model

[meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

## Reference

- [SG-Lang](https://docs.sglang.ai/)
- [SG-Lang Github](https://github.com/sgl-project/sglang?tab=readme-ov-file)
- [SGLang 推理引擎技术解析](https://zhuanlan.zhihu.com/p/30886364337)
- [llm-compressor](https://github.com/vllm-project/llm-compressor/tree/main)
- [TorchTune](https://github.com/pytorch/torchtune)
- [TorchTune/configs](https://github.com/pytorch/torchtune/tree/main/recipes/configs)
