# Sample Code : https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w8a8_int8/README.md
# SparseGPT: https://github.com/vllm-project/llm-compressor/blob/main/examples/sparse_2of4_quantization_fp8/llama3_8b_2of4.py
# %%
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# %%

from datasets import load_dataset
import torch

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048
# %%
# Load dataset.
ds = load_dataset(
    "wikitext", "wikitext-2-raw-v1", split=f"train[:{NUM_CALIBRATION_SAMPLES}]"
)
ds = ds.shuffle(seed=42)

# Filter out samples with empty text
ds = ds.filter(lambda example: len(example["text"]) > 0)


print(ds)


# Tokenize the data (be careful with bos tokens - we need add_special_tokens=False since the chat_template already added it).
def tokenize(sample):
    tokenized = tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )
    return {
        "input_ids": torch.tensor(tokenized["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(tokenized["attention_mask"], dtype=torch.long),
    }


# %%

ds = ds.map(tokenize, remove_columns=ds.column_names)

ds.set_format(
    type="torch",
    # columns=["input_ids", "attention_mask"],
    # output_all_columns=True,
)


# %%
print(ds)
print(ds[0])
print(ds[0]["input_ids"])
print(type(ds[0]["input_ids"]))

# %%

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.obcq import SparseGPTModifier

# Configure the quantization algorithms to run.
recipe = [
    SparseGPTModifier(
        sparsity=0.5,
        mask_structure="2:4",
        sequential_update=True,
        targets=[r"re:model.layers.\d*$"],
    ),
    GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
]


# %%

model_dir = MODEL_ID.split("/")[1] + "-W4A16-Dynamic-Per-Token-SparseGPT"


# Apply quantization.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=len(ds),
    output_dir=model_dir,  # No need to save the model yet
)
# %%
# Save to disk compressed.
SAVE_DIR = f"{model_dir}-V2"
tokenizer.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR, save_compressed=True)

# %%
