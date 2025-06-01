# Sample Code : https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w8a8_int8/README.md
import torch
from datasets import load_dataset

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

from transformers import AutoTokenizer, AutoModelForCausalLM
import click

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048


def build_dataset(tokenizer):

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
            "attention_mask": torch.tensor(
                tokenized["attention_mask"], dtype=torch.long
            ),
        }

    # Load dataset.
    ds = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split=f"train[:{NUM_CALIBRATION_SAMPLES}]"
    )
    ds = ds.shuffle(seed=42)

    # Filter out samples with empty text
    ds = ds.filter(lambda example: len(example["text"]) > 0)

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    ds.set_format(
        type="torch",
        # columns=["input_ids", "attention_mask"],
        # output_all_columns=True,
    )

    return ds


def get_output_path(model_name: str):
    if "/" in model_name:
        model_name = model_name.split("/")[-1]

    return f"{model_name}-W8A8-Dynamic-Per-Token"


@click.command()
@click.option(
    "--model_name",
    default="meta-llama/Llama-3.2-3B-Instruct",
)
def main(model_name: str):
    print("Loading model and tokenizer...")
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = build_dataset(tokenizer)

    print("Dataset loaded and tokenized.")
    print(ds)
    print("Starting quantization...")

    # Configure the quantization algorithms to run.
    recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
    ]

    output_path = get_output_path(model_name)
    # Apply quantization.
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=len(ds),
        output_dir=output_path,  # No need to save the model yet
    )

    print(model)
    print(f"Quantization completed. Model saved to {output_path}.")
    return


if __name__ == "__main__":
    main()
