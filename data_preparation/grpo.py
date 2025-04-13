import argparse
import os
from typing import Any, Callable

import datasets
import pandas as pd


# Define the prompt template
prompt_template = """You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. Write a response that answers the following question. 
Your response MUST and ONLY contain two parts: thinking and answer. Thinking part provides the reasoning of your response and MUST start with <think> and end with </think>. Answer part is the summary of the reasoning with the final answer and MUST start with <answer> and end with </answer>. Your reponse MUST contain <think>, </think>, <answer>, and </answer>. 
YOUR RESPONSE MUST STRICTLY BE OF THE FOLLOWING FORMAT:
<think>
Your thinking here.
</think>
<answer>
Summary of thinking and final answer here.
</answer>

Question:
{question}
Your final choice MUST be included in the box \\boxed{{}}. For example, \\boxed{{A}} or \\boxed{{B}}.
"""


def parse_args() -> tuple[str, str, str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", type=str, required=True, choices=["medqa", "medmcqa"])
    parser.add_argument("--split", type=str, required=True, choices=["train", "dev"])
    parser.add_argument("--data_path", type=str, required=True, help="csv path for the processed data")
    parser.add_argument("--save_path", type=str, required=True, help="path to save the parquet file")
    args = parser.parse_args()
    return args.data_source, args.split, args.data_path, args.save_path


def get_create_prompt(prompt_template: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def create_prompt(row: dict[str, Any]) -> dict[str, Any]:
        row["prompt_question"] = prompt_template.format(question=row["question"])
        return row
    return create_prompt


def make_map_fn(data_source, split):
    def process_fn(example, idx):
        question = example.pop("question").strip()
        prompt_question = example.pop("prompt_question").strip()
        ground_truth = example.pop("ground_truth").strip()
        answer = example.pop("correct_answer").strip()

        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": prompt_question,
            }],
            "ability": "medical",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer,
                "question": question,
            }
        }
        return data
    return process_fn


def main():
    data_source, split, data_path, save_path = parse_args()

    df = pd.read_csv(data_path)
    df = df.map(get_create_prompt(prompt_template), axis=1)
    df_dataset = datasets.Dataset.from_pandas(df)
    df_dataset = df_dataset.map(function=make_map_fn(data_source, split), with_indices=True)

    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    df_dataset.to_parquet(save_path)


if __name__ == "__main__":
    main()