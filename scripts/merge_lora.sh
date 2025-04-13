python evaluation/merge_lora.py \
    --base_model_path Qwen/Qwen2.5-3B-Instruct \
    --lora_model_path model_checkpoints/sft/Qwen--Qwen2.5-3B-Instruct-medqa-no-reasoning-sft-epoch20-batch256/global_step_340 \
    --merged_model_save_path model_checkpoints/sft/Qwen--Qwen2.5-3B-Instruct-medqa-no-reasoning-sft-epoch20-batch256/global_step_340_full_params

python evaluation/merge_lora.py \
    --base_model_path Qwen/Qwen2.5-3B-Instruct \
    --lora_model_path model_checkpoints/sft/Qwen--Qwen2.5-3B-Instruct-deepseek-medqa-distil-epoch20-batch256/global_step_221 \
    --merged_model_save_path model_checkpoints/sft/Qwen--Qwen2.5-3B-Instruct-deepseek-medqa-distil-epoch20-batch256/global_step_221_full_params