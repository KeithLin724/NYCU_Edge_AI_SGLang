from transformers import AutoModelForCausalLM, AutoTokenizer
import click


def get_model_name(quant: bool):

    if not quant:
        return (
            "hlhsiao/llama-3.2-1b-KD-V1",
            "meta-llama/Llama-3.2-1B-Instruct",
            "llama-3.2-1b-KD-V1",
        )

    return (
        "KYLiN724/llama-3.2-1b-KD-V1-W8A8-Dynamic-Per-Token",
        "KYLiN724/llama-3.2-1b-KD-V1-W8A8-Dynamic-Per-Token",
        "llama-3.2-1b-KD-V1-W8A8-Dynamic-Per-Token",
    )


@click.command()
@click.option(
    "--quant",
    is_flag=True,
    help="Download quantization version of the model. Default is False (float32 version).",
)
def main(quant: bool):

    model_name, tokenizer_name, save_path = get_model_name(quant)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"Model and tokenizer saved to {save_path}")


if __name__ == "__main__":
    main()
