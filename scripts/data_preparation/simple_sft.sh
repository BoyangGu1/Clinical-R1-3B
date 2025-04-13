python data_preparation/simple_sft.py \
    --data_source medqa \
    --split train \
    --data_path data/processed/medqa_dev.csv \
    --save_path data/verl/medqa_simple_sft/train.parquet

python data_preparation/simple_sft.py \
    --data_source medqa \
    --split val \
    --data_path data/processed/medqa_dev.csv \
    --save_path data/verl/medqa_simple_sft/val.parquet