# Model Distillation Project

This project implements knowledge distillation from a larger language model (teacher) to a smaller model (student) using the TorchTune framework.

> Pre-build model : <https://huggingface.co/hlhsiao/llama-3.2-1b-KD-V1>

## Environment Setup

1. Create a new conda environment using the provided `environment.yml`:

```bash
conda env create -f environment.yml
```

2. Activate the environment:

```bash
conda activate torchtune
```

## Model Preparation

Before running the distillation process, you need to download both the teacher and student models. Make sure to download them to the correct paths as specified in the configuration file:

1. Download the student model (1B parameters):

```bash
tune download meta-llama/Llama-3.2-1B-Instruct \
    --output-dir <YOUR_STUDENT_MODEL_PATH> \
    --ignore-patterns "original/consolidated.00.pth"
```

2. Download the teacher model (3B parameters):

```bash
tune download meta-llama/Llama-3.2-3B-Instruct \
    --output-dir <YOUR_TEACHER_MODEL_PATH> \
    --ignore-patterns "original/consolidated.00.pth"
```

## Configuration Setup

Before running the distillation process, you need to modify the following paths in `8B_to_1B_KD_lora_single_device.yaml`:

1. Set your output directory:

```yaml
output_dir: <YOUR_OUTPUT_DIR>
```

2. Update the model paths:

```yaml
tokenizer:
  path: <YOUR_STUDENT_MODEL_PATH>/original/tokenizer.model

checkpointer:
  checkpoint_dir: <YOUR_STUDENT_MODEL_PATH>

teacher_checkpointer:
  checkpoint_dir: <YOUR_TEACHER_MODEL_PATH>
```

Replace the placeholders with your actual paths:

- `<YOUR_OUTPUT_DIR>`: Directory where the distilled model will be saved
- `<YOUR_STUDENT_MODEL_PATH>`: Path where you downloaded the student model
- `<YOUR_TEACHER_MODEL_PATH>`: Path where you downloaded the teacher model

## Running Knowledge Distillation

After setting up the environment and downloading the models, you can run the knowledge distillation process using the provided configuration file:

```bash
tune run knowledge_distillation_single_device --config 8B_to_1B_KD_lora_single_device.yaml
```

## Merging LoRA Adapters

After the distillation process is complete, you need to merge the LoRA adapters with the base model(Llama-3.2-1B-Instruct). Before running the merge script, modify the paths in `merge_lora.py`:

```python
# Load PEFT adapters
model = PeftModel.from_pretrained(base_model, "<YOUR_OUTPUT_DIR>/epoch_0")
model = model.merge_and_unload()

tokenizer.save_pretrained("<YOUR_MERGED_MODEL_PATH>")
model.save_pretrained("<YOUR_MERGED_MODEL_PATH>")
```

Replace the placeholders:

- `<YOUR_OUTPUT_DIR>`: Same as the output directory you set in the configuration file
- `<YOUR_MERGED_MODEL_PATH>`: Directory where you want to save the merged model

Then run the merge command:

```bash
python merge_lora.py
```

This will merge the trained LoRA adapters with the base student model, creating a new model that incorporates the knowledge from the teacher model.

## Model Evaluation

To evaluate the performance of the distilled model, including throughput and perplexity metrics, run:

```bash
python result.py
```

This script will:

- Measure the model's throughput (tokens/second)
- Calculate perplexity on the evaluation dataset

## Configuration

The distillation process is configured through the `8B_to_1B_KD_lora_single_device.yaml` file. Key parameters include:

- Model configurations for both teacher and student models
- Training parameters (learning rate, batch size, etc.)
- LoRA settings for efficient fine-tuning
- Dataset configuration
- Optimization settings

## Notes

- The distillation process uses LoRA (Low-Rank Adaptation) for efficient fine-tuning
- The configuration is optimized for single-device training
- Make sure you have sufficient GPU memory for the distillation process
- The output directory can be modified in the configuration file according to your needs
- After merging the LoRA adapters, the final model will be saved in the specified output directory
- The evaluation results will help you understand the trade-off between model size and performance
