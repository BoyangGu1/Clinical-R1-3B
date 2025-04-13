#!/bin/bash
save_dirs=("cot" "simple-sft" "deepseek-distil" "grpo-only" "clinical-r1-3b")

gpus="0,1,2,3,4,5,6,7"

# Loop through each string in the list
for i in $(seq 0 4); do
    save_dir=${save_dirs[$i]}
    if [ "$save_dir" == "cot" ] || [ "$save_dir" == "simple-sft" ]; then
        CUDA_VISIBLE_DEVICES=$gpus python evaluation/model_response_evaluation.py \
            --response_dir inference/model_response_generation/medqa/$save_dir \
            --gpus $gpus \
            --dataset_path data/processed/medqa_test.csv \
            --evaluation_response_save_dir inference/model_response_evaluation/medqa/$save_dir \
            --evaluation_results_save_path "evaluation_results/accuracy/medqa/$save_dir.json"

        CUDA_VISIBLE_DEVICES=$gpus python evaluation/model_response_evaluation.py \
            --response_dir inference/model_response_generation/medmcqa/$save_dir \
            --gpus $gpus \
            --dataset_path data/processed/medmcqa_dev.csv \
            --evaluation_response_save_dir inference/model_response_evaluation/medmcqa/$save_dir \
            --evaluation_results_save_path "evaluation_results/accuracy/medmcqa/$save_dir.json"
    else
        CUDA_VISIBLE_DEVICES=$gpus python evaluation/model_response_evaluation.py \
            --response_dir inference/model_response_generation/medqa/$save_dir \
            --gpus $gpus \
            --dataset_path data/processed/medqa_test.csv \
            --answer_extraction_regex '\\boxed{(.*?)}' \
            --evaluation_results_save_path "evaluation_results/accuracy/medqa/$save_dir.json"

        CUDA_VISIBLE_DEVICES=$gpus python evaluation/model_response_evaluation.py \
            --response_dir inference/model_response_generation/medmcqa/$save_dir \
            --gpus $gpus \
            --dataset_path data/processed/medmcqa_dev.csv \
            --answer_extraction_regex '\\boxed{(.*?)}' \
            --evaluation_results_save_path "evaluation_results/accuracy/medmcqa/$save_dir.json"
    fi
done