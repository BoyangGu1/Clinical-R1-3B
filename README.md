# Clinical-R1-3B
Clinical-R1-3B: A reinforcement learning-based model inspired by DeepSeek-R1, designed to enhance medical question-answering and reasoning capabilities.

This repository is based on [veRL](https://github.com/volcengine/verl) and [Qwen-2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).

All datasets and models are available on the [HF Collection]().

## Installation

```
conda env create -f environment.yaml -n verl
conda activate verl
pip install -e .
```

## Data Preparation

```
./scripts/data_preparation/raw2csv.sh
./scripts/data_preparation/grpo.sh
./scripts/data_preparation/simple_sft.sh
```

For distillation (cold start) data preparation, you need to fill in your Deepseek API key and conda environment path into `./scripts/data_preparation/deepseek_api_call.sh`, and run
```
./scripts/data_preparation/deepseek_api_call.sh
```
After all screens are done with API calls, run
```
./scripts/data_preparation/distil_sft.sh
```

## Training
### Simple SFT (without reasoning)
```
./scripts/sft/run_qwen_3b_peft_simple_sft.sh
```
### Cold Start
```
./scripts/sft/run_qwen_3b_peft_deepseek_distil.sh
```
Before running GRPO-related training, we need to process the merge the peft models:
```
./scripts/merge_lora.sh
```
### GRPO Only
```
./scripts/grpo/run_qwen-3B_seq_balance_medqa.sh
```
### Clinical-R1-3B
```
./scripts/grpo/run_qwen-3B_after-deepseek-distil-on-medqa_seq_balance_medqa.sh
```

## Evaluation
```
./scripts/evaluation/model_response_generation.sh
./scripts/evaluation/model_response_evaluation.sh
./scripts/evaluation/cognitive_behavior_evaluation.sh
```