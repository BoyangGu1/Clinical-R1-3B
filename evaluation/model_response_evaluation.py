import argparse
import json
import os
import re
import shutil
from typing import Any, Callable, Optional

import ray
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams


user_prompt = """You are a multiple choice question marking assistant. You will be given a multiple choice question and an answer candidate. Your task is to extract the answer choice from the answer. DO NOT EVALUATE THE CORRECTNESS OF THE ANSWER CANDIDATE. ONLY EXTRACT THE ANSWER CHOICE FROM THE ANSWER CANDIDATE GIVEN.

Here comes the actual question and answer:
Question: 
{question}

Answer Candidate: 
{model_answer}

YOUR RESPONSE MUST BE IN THE FOLLOWING FORMAT:
Response:
Let us think step by step: {{your reasoing of which part of the answer candidate indicates their final choice}}
Answer Candidate Choice: {{answer candidate choice, such as A, B, C, or D}}
"""


def parse_args() -> tuple[str, str, str, Optional[str], Optional[str], str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_dir", required=True, type=str, help="location for generated responses, should be a directory")
    parser.add_argument("--gpus", required=False, default="0,1,2,3,4,5,6,7", type=str, help="the gpu indices for this script to run on")
    parser.add_argument("--dataset_path", required=True, type=str, help="location for document-summary pairs, should be a csv file")
    parser.add_argument("--answer_extraction_regex", required=False, default=None, type=str, help="regex form for extracting model answers from model responses")
    parser.add_argument("--evaluation_response_save_dir", required=False, default=None, type=str, help="location for saving generated responses for evaluation, should be a directory")
    parser.add_argument("--evaluation_results_save_path", required=True, type=str, help="location for saving evaluation results, should be a json file")
    args = parser.parse_args()
    return args.response_dir, args.gpus, args.dataset_path, args.answer_extraction_regex, args.evaluation_response_save_dir, args.evaluation_results_save_path


def get_LLMPredictor(tensor_parallel_size, sampling_params, save_dir):

    class LLMPredictor:

        def __init__(self):
            # Create an LLM.
            self.llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
                        tensor_parallel_size=tensor_parallel_size)
            self.tokenizer = self.llm.get_tokenizer()
            self.sampling_params = sampling_params
            self.save_dir = save_dir

        def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, list]:
            # Generate texts from the prompts.
            # The output is a list of RequestOutput objects that contain the prompt,
            # generated text, and other information.
            conversations = []
            for text in batch["text"]:
                conversation = self.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": text},
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                conversations.append(conversation)
            # change that so conversation takes the batch[text]
            outputs = self.llm.generate(conversations, self.sampling_params)

            prompt: list[str] = []
            generated_text: list[str] = []
            for output in outputs:
                prompt.append(output.prompt)
                response = " ".join([o.text for o in output.outputs])
                generated_text.append("\n".join(response.split("\n")))

            # Save the generated text to a file named after the pair_id.
            pair_ids = batch["question_index"]
            for pair_id, one_generated_text in zip(pair_ids, generated_text):
                filename = os.path.join(self.save_dir, f"{pair_id}.txt")
                with open(filename, "w") as f:
                    f.write(one_generated_text)

            return {
                "prompt": prompt,
                "generated_text": generated_text,
            }
        
    return LLMPredictor


def is_valid_cuda_visible_devices(cuda_str: str) -> bool:
    if cuda_str == "":
        return True  # An empty string is a valid value
    devices = cuda_str.split(",")
    try:
        # Convert to integers and check for duplicates
        device_numbers = list(map(int, devices))
    except ValueError:
        return False  # Non-integer value present
    # Check for non-negative integers and duplicates
    if any(d < 0 for d in device_numbers) or len(device_numbers) != len(set(device_numbers)):
        return False
    return True


def get_create_prompt(user_prompt: str, response_dir: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def create_prompt(row: dict[str, Any]) -> dict[str, Any]:
        with open(os.path.join(response_dir, f"{row['question_index']}.txt"), "r") as f:
            response = f.read()
        row["response"] = response
        row["text"] = user_prompt.format(question=row["question"], model_answer=row["response"])
        return row
    return create_prompt


def get_accuracy(dataset_df: pd.DataFrame, response_dir: str, answer_extraction_regex: str):
    correct_count = 0
    total_count = 0
    for i, row in dataset_df.iterrows():
        question_index = row["question_index"]
        with open(os.path.join(response_dir, f"{question_index}.txt"), "r") as f:
            model_answer_text = f.read().strip()
        model_answers = re.findall(answer_extraction_regex, model_answer_text)
        if model_answers and len(model_answers) == 1 and model_answers[0].strip().upper() in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]:
            total_count += 1
            model_answer = model_answers[0].strip()
            correct_answer = row["correct_answer"]
            if model_answer == correct_answer:
                correct_count += 1
    accuracy = correct_count / total_count
    valid_count_ratio = total_count / len(dataset_df)
    return correct_count, total_count, accuracy, valid_count_ratio


def main():
    response_dir, gpus, dataset_path, answer_extraction_regex, evaluation_response_save_dir, evaluation_results_save_path = parse_args()
    if (answer_extraction_regex is None and evaluation_response_save_dir is None) or (answer_extraction_regex is not None and evaluation_response_save_dir is not None):
        raise ValueError("You should input only either a valid answer_extraction_regex or evaluation_response_save_dir")
    
    if evaluation_response_save_dir is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        assert is_valid_cuda_visible_devices, "Invalid CUDA_VISIBLE_DEVICES value"

        # Create a sampling params object.
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=4096)
        # Set tensor parallelism per instance.
        tensor_parallel_size = 1
        # Set number of instances. Each instance will use tensor_parallel_size GPUs.
        num_instances = len(gpus.split(","))

        df = pd.read_csv(dataset_path)
        ds = ray.data.from_pandas(df)
        ds = ds.map(get_create_prompt(user_prompt, response_dir))
        ds = ds.repartition(128)

        resources_kwarg: dict[str, Any] = {}
        resources_kwarg["num_gpus"] = 1

        llm_predictor = get_LLMPredictor(tensor_parallel_size, sampling_params, evaluation_response_save_dir)
        os.makedirs(evaluation_response_save_dir, exist_ok=True)

        ds = ds.map_batches(
            llm_predictor,
            concurrency=num_instances,
            batch_size=32,
            **resources_kwarg,
        )

        os.makedirs("temp_dir", exist_ok=True)
        ds.write_parquet("temp_dir")
        shutil.rmtree("temp_dir")

        correct_count, total_count, accuracy, valid_count_ratio = get_accuracy(df, response_dir, r"Answer Candidate Choice:\s*(\w+)[\.]?")
        evaluation_results_dict = {"correct_count": correct_count,
                                   "total_count": total_count,
                                   "accuracy": accuracy,
                                   "valid_count_ratio": valid_count_ratio}
        with open(evaluation_results_save_path, "w") as f:
            json.dump(evaluation_results_dict, f)

    elif answer_extraction_regex is not None:
        df = pd.read_csv(dataset_path)
        correct_count, total_count, accuracy, valid_count_ratio = get_accuracy(df, response_dir, r"Answer Candidate Choice:\s*(\w+)[\.]?")
        evaluation_results_dict = {"correct_count": correct_count,
                                   "total_count": total_count,
                                   "accuracy": accuracy,
                                   "valid_count_ratio": valid_count_ratio}
        with open(evaluation_results_save_path, "w") as f:
            json.dump(evaluation_results_dict, f)

    else:
        raise ValueError("You should input only either a valid answer_extraction_regex or evaluation_response_save_dir")


if __name__ == "__main__": 
    main()