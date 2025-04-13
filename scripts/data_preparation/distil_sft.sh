python data_preparation/distil_sft.py \
    --data_source medqa \
    --split train \
    --data_path data/processed/medqa_train_sft.csv \
    --distil_response_save_dir data/distil/medqa/deepseek-r1 \
    --save_path data/verl/medqa_deepseek_distil_sft/train.parquet

python data_preparation/distil_sft.py \
    --data_source medqa \
    --split val \
    --data_path data/processed/medqa_train_sft.csv \
    --distil_response_save_dir data/distil/medqa/deepseek-r1 \
    --save_path data/verl/medqa_deepseek_distil_sft/val.parquet