import argparse
import os
import json

import pandas as pd


def parse_args() -> tuple[str, str, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="medqa", type=str, choices=["medqa", "medmcqa"])
    parser.add_argument("--save_dir", default=os.path.join("data", "processed"), type=str)
    parser.add_argument("--sft_grpo_split", action="store_true", type=bool)
    args = parser.parse_args()
    return args.dataset_name, args.save_dir, args.sft_grpo_split


def medqa_prep(split="test"):
    # Read medqa_test.jsonl and fit the prompt template
    with open(os.path.join("data", "raw", f"medqa_{split}.jsonl"), "r") as file:
        lines = file.readlines()
    
    data = []
    for i, line in enumerate(lines):
        record = json.loads(line.strip())
        record["question_index"] = i  # Adding index for answer-question pair identification
        answer_idx = record["answer_idx"]
        record["correct_answer"] = answer_idx
        record["choice_type"] = "single"  # Handle choice type (single or multi)
        record["question"] = record["question"].strip() + "\n" + "\n".join([f"{key.strip()}: {value.strip()}" for key, value in record["options"].items()])
        record["ground_truth"] = record["answer"] + "<SEP>" + split + f"{record['question_index']}<SEP>" + "<SEP>".join(record["options"].values())

        data.append(record)
    
    df = pd.DataFrame(data)
    return df


def medmcqa_prep(split="dev"):
    # Read medmcqa_test.json and fit the prompt template
    with open(os.path.join("data", f"medmcqa_{split}.json"), "r") as file:
        lines = file.readlines()
    
    data = []
    for i, line in enumerate(lines):
        record = json.loads(line.strip())

        # Prepare the options dictionary based on available keys
        options = {
            "A": record.get("opa", ""),
            "B": record.get("opb", ""),
            "C": record.get("opc", ""),
            "D": record.get("opd", "")
        }
        options = {key: value for key, value in options.items() if value}
        record["options"] = options

        answer_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
        numerical_answer = record.get("cop", "")
        letter_answer = answer_map.get(numerical_answer, "")
        record["correct_answer"] = letter_answer

        record["question_index"] = i  # Adding index for answer-question pair identification
        record["choice_type"] = record.get("choice_type", "single")  # Handle choice type (single or multi)
        choice_type_line = ""
        if record["choice_type"] == "multi":
            choice_type_line = "This question is a multi-choice question. There could be more than one correct answer.\n"
        record["question"] = record["question"].strip() + "\n" + choice_type_line + "\n".join([f"{key.strip()}: {value.strip()}" for key, value in record["options"].items()]).strip()
        record["ground_truth"] = record["answer"] + "<SEP>" + split + f"{record['question_index']}<SEP>" + "<SEP>".join(record["options"].values())

        data.append(record)
    
    df = pd.DataFrame(data)
    return df


def main():
    dataset_name, save_dir, sft_grpo_split = parse_args()
    
    os.makedirs(save_dir)
    if dataset_name == "medqa":
        for split in ["train", "dev", "test"]:
            df = medqa_prep(split="dev")
            df.to_csv(os.path.join("data", "processed", f"{dataset_name}_{split}.csv"), index=False)

            if split == "train" and sft_grpo_split:
                sft_df = df[:(len(df) // 2)]
                grpo_df = df[(len(df) // 2):]
                sft_df.to_csv(os.path.join("data", "processed", f"{dataset_name}_{split}_sft.csv"), index=False)
                grpo_df.to_csv(os.path.join("data", "processed", f"{dataset_name}_{split}_grpo.csv"), index=False)

    elif dataset_name == "medmcqa":
        for split in ["train", "dev"]:
            df = medqa_prep(split="dev")
            df.to_csv(os.path.join("data", "processed", f"{dataset_name}_{split}.csv"), index=False)

            if split == "train" and sft_grpo_split:
                sft_df = df[:(len(df) // 2)]
                grpo_df = df[(len(df) // 2):]
                sft_df.to_csv(os.path.join("data", "processed", f"{dataset_name}_{split}_sft.csv"), index=False)
                grpo_df.to_csv(os.path.join("data", "processed", f"{dataset_name}_{split}_grpo.csv"), index=False)

    else:
        raise ValueError(f"{dataset_name} is not supported.")
    

if __name__ == "__main__":
    main()