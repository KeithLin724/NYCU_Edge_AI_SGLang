# Build LLama3.2 1B Instruction Model with Quantization (W8A8 int8)

> Make sure you have set up the environment and installed all dependencies as described in the main `README.md`.

## Step 1: Download the base model

You can obtain the student model in one of two ways:

1. **Direct download from Hugging Face:**
   - **Model link:** [hlhsiao/llama-3.2-1b-KD-V1](https://huggingface.co/hlhsiao/llama-3.2-1b-KD-V1)
   - **Download with script:**

     ```sh
     python get_preprocess_model.py
     ```

2. **Build it yourself:**
   - Follow the instructions in [build_small_model/README.md](./build_small_model/README.md)

## Step 2: Quantize the model to INT8

After obtaining the FP16 student model, quantize it to INT8 by running:

```sh
# Quantize the model (hlhsiao/llama-3.2-1b-KD-V1)
python compress_to_int8.py --model_name hlhsiao/llama-3.2-1b-KD-V1

# Or, quantize a custom model by specifying the model name or path
# python compress_to_int8.py --model_name <model_name_or_path>
```

The quantized model will be saved to a new directory for further use.
