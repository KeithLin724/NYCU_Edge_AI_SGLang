# Build LLama3.2 1B Instruction Model with Quantization (W8A8 int8)

## Step 1: Get the 1B Model

### Option 1: Download the Pre-built Model

```sh
python get_preprocess_model.py
```

### Option 2: Build the Model Yourself

See instructions: [build_small_model/README.md](./build_small_model/README.md)

---

## Step 2: Run the Quantization Script

After obtaining the model, run the following command to quantize it to W8A8 int8:

```sh
python compress_int_1b.py
```

---

> Make sure you have set up the environment and installed all dependencies as described in the main `README.md`.
