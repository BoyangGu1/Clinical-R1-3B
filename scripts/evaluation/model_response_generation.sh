#!/bin/bash
model_paths=("Qwen/Qwen2.5-3B-Instruct" \
             "model_checkpoints/sft/Qwen--Qwen2.5-3B-Instruct-medqa-no-reasoning-sft-epoch20-batch256/global_step_340_full_params" \
             "model_checkpoints/sft/Qwen--Qwen2.5-3B-Instruct-deepseek-medqa-distil-epoch20-batch256/global_step_221_full_params" \
             "model_checkpoints/Qwen--Qwen2.5-3B-Instruct-medqa-grpo-epoch20-batch1024/actor/global_step_35" \
             "model_checkpoints/grpo/Qwen--Qwen2.5-3B-Instruct-deepseek-distil-then-grpo-medqa-epoch20-batch512/actor/global_step_105")

save_dirs=("cot" "simple-sft" "deepseek-distil" "grpo-only" "clinical-r1-3b")

gpus="0,1,2,3,4,5,6,7"

# Loop through each string in the list
for i in $(seq 0 4); do
    model_path=${model_paths[$i]}
    save_dir=${save_dirs[$i]}
    if [ "$model_path" == "model_checkpoints/sft/Qwen--Qwen2.5-3B-Instruct-medqa-no-reasoning-sft-epoch20-batch256/global_step_340_full_params" ]; then
        user_prompt_type="regular_user_prompt"
    else
        user_prompt_type="thinking_user_prompt"
    fi

    CUDA_VISIBLE_DEVICES=$gpus python evaluation/model_response_generation.py \
        --model_path Qwen/Qwen2.5-3B-Instruct \
        --gpus $gpus \
        --dataset_path data/processed/medqa_test.csv \
        --save_dir inference/model_response_generation/medqa/$save_dir \
        --user_prompt_type $user_prompt_type

    CUDA_VISIBLE_DEVICES=$gpus python evaluation/model_response_generation.py \
        --model_path Qwen/Qwen2.5-3B-Instruct \
        --gpus $gpus \
        --dataset_path data/processed/medmcqa_dev.csv \
        --save_dir inference/model_response_generation/medmcqa/$save_dir \
        --user_prompt_type $user_prompt_type
done