#!/bin/bash
save_dirs=("cot" "simple-sft" "deepseek-distil" "grpo-only" "clinical-r1-3b")

gpus="0,1,2,3,4,5,6,7"

# Loop through each string in the list
for i in $(seq 0 4); do
    save_dir=${save_dirs[$i]}

    CUDA_VISIBLE_DEVICES=$gpus python evaluation/cognitive_behavior_evaluation.py \
        --gpus $gpus \
        --dataset_path data/processed/medqa_test.csv \
        --response_dir inference/model_response_generation/medqa/$save_dir \
        --behavior_response_save_dir inference/cognitive_behaviors/medqa/$save_dir \
        --behavior_results_save_path "evaluation_results/cognitive_behavior_counts/medqa/$save_dir.json"

    CUDA_VISIBLE_DEVICES=$gpus python evaluation/cognitive_behavior_evaluation.py \
        --gpus $gpus \
        --dataset_path data/processed/medmcqa_dev.csv \
        --response_dir inference/model_response_generation/medmcqa/$save_dir \
        --behavior_response_save_dir inference/cognitive_behaviors/medmcqa/$save_dir \
        --behavior_results_save_path "evaluation_results/cognitive_behavior_counts/medmcqa/$save_dir.json"
done