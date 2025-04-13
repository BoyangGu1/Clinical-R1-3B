/data/bob_files/verl/model_checkpoints/sft/Qwen--Qwen2.5-3B-Instruct-deepseek-medqa-distil-epoch20-batch256/global_step_221_full_params

/data/bob_files/verl/model_checkpoints/sft/Qwen--Qwen2.5-3B-Instruct-no-reasoning-epoch20-batch256-medqa/global_step_340_full_params

/data/bob_files/verl/model_checkpoints/grpo/Qwen--Qwen2.5-3B-Instruct-medqa-epoch20-batch1024-exp2/actor/global_step_35

/data/bob_files/verl/model_checkpoints/grpo/Qwen--Qwen2.5-3B-Instruct-deepseek-distil-then-grpo-medqa-epoch20-batch512/actor/global_step_105



scp -r bobgu@10.250.30.33:/data/bob_files/verl/model_checkpoints/sft/Qwen--Qwen2.5-3B-Instruct-deepseek-medqa-distil-epoch20-batch256/global_step_221_full_params model_checkpoints/cold_start


scp -r bobgu@10.250.30.33:/data/bob_files/verl/model_checkpoints/sft/Qwen--Qwen2.5-3B-Instruct-no-reasoning-epoch20-batch256-medqa/global_step_340_full_params model_checkpoints/simple_sft


scp -r bobgu@10.250.30.33:/data/bob_files/verl/model_checkpoints/grpo/Qwen--Qwen2.5-3B-Instruct-medqa-epoch20-batch1024-exp2/actor/global_step_35 model_checkpoints/grpo_only


scp -r bobgu@10.250.30.33:/data/bob_files/verl/model_checkpoints/grpo/Qwen--Qwen2.5-3B-Instruct-deepseek-distil-then-grpo-medqa-epoch20-batch512/actor/global_step_105 model_checkpoints/clinical-r1-3b