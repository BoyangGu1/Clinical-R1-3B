# Tested with 2 & 4 GPUs
set -x

nproc_per_node=8
save_path=model_checkpoints/sft/Qwen--Qwen2.5-3B-Instruct-medqa-no-reasoning-sft-epoch20-batch256

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=data/verl/medqa_simple_sft/train.parquet \
    data.val_files=data/verl/medqa_simple_sft/val.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=1e-4 \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.max_length=1024 \
    data.truncation=right \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=Qwen/Qwen2.5-3B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=clinical-r1-sft \
    trainer.experiment_name=medqa-no-reasoning-sft-qwen-2.5-3b-instruct \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=20 \
    trainer.default_hdfs_dir=null \
    +trainer.cross_entropy_loss_label_smoothing=0 \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    model.target_modules=all-linear 2>&1 | tee logging/verl-medqa-no-reasoning-sft-qwen3b-epoch20-batch256.log

    # Or you can do this:
    # model.target_modules=[q_proj,v_proj] \
