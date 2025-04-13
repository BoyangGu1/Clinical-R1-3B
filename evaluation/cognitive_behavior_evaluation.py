import argparse
import os
import shutil
import re
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import ray

from vllm import LLM, SamplingParams
import json


# 1. Answer-verification steps
verification_prompt = """Here is a chain-of-reasoning that a Language Model generated while trying to solve a medical-domain multiple choice question. The goal is to select the correct answer from all given options. 
The question is:
{question}
The correct answer is: {correct_answer}
The chain-of-reasoning the model used is: 
{response}. 

Evaluate whether the chain-of-reasoning contains any answer-verification steps. An example of an answer-verification step is: "This patient's symptoms align with hypothyroidism, which is consistent with the lab results showing elevated TSH levels.". We want to mark instances where the chain-of-reasoning explicitly checks the current result against the given conditions. 

If you find any answer-verification steps, please count them and provide the count as between the tags <count> </count>. If the chain-of-reasoning does not contain any answer-verification steps, please provide a count of 0 as <count>0</count>."""

# 2. Backtracking behavior
backtracking_prompt = """Here is a chain-of-reasoning that a Language Model generated while trying to solve a medical-domain multiple choice question. The goal is to select the correct answer from all given options. 
The question is:
{question}
The correct answer is: {correct_answer}
The chain-of-reasoning the model used is: 
{response}. 

Evaluate whether the chain-of-reasoning contains any backtracking behavior, where the model realizes a path won't work and explicitly goes back to try a different approach. An example of backtracking is: "Let me try again" or "we need to try a different sequence". We want to mark instances where the chain-of-reasoning is abandoned and the model backtracks to a previous thought. 

Count the number of distinct backtracking instances and provide the count between the tags <count> </count>. If the chain-of-reasoning does not contain any backtracking behavior, please provide a count of 0 as <count>0</count>."""

# 3. Subgoal setting
subgoal_prompt = """Here is a chain-of-reasoning that a Language Model generated while trying to solve a medical-domain multiple choice question. The goal is to select the correct answer from all given options. 
The question is:
{question}
The correct answer is: {correct_answer}
The chain-of-reasoning the model used is: 
{response}. 

Evaluate whether the chain-of-reasoning contains any explicit subgoal setting, where the model breaks down the problem into smaller, intermediate goals. An example of subgoal setting is: "First, I'll assess the patient's vital signs to rule out immediate life-threatening conditions; then, I'll order relevant laboratory tests for further evaluation.".

Count the number of distinct subgoals set and provide the count between the tags <count> </count>. If the chain-of-reasoning does not contain any subgoal setting, please provide a count of 0 as <count>0</count>."""

# 4. Backward-chaining
backward_chaining_prompt = """Here is a chain-of-reasoning that a Language Model generated while trying to solve a medical-domain multiple choice question. The goal is to select the correct answer from all given options. 
The question is:
{question}
The correct answer is: {correct_answer}
The chain-of-reasoning the model used is: 
{response}. 

Evaluate whether the chain-of-reasoning contains any backward-chaining behavior, where the model begins with a specific hypothesis or diagnosis and works backward to determine if available data and symptoms support that conclusion. An example of backward-chaining is: "Given the patient's elevated serum calcium levels, let's consider possible causes such as hyperparathyroidism or malignancy.".

Count the number of distinct backward-chaining instances and provide the count between the tags <count> </count>. If the chain-of-reasoning does not contain any backward-chaining behavior, please provide a count of 0 as <count>0</count>."""


def parse_args() -> tuple[str, str, Optional[str], Optional[str], str, str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker_model_path", required=False, type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="the name of the model to evaluate the congnitive behaviors")
    parser.add_argument("--gpus", required=True, type=str, help="the gpu indices for this script to run on")
    parser.add_argument("--dataset_name", required=False, type=str, choices=["medqa", "medmcqa", "medxpertqa"], default=None, help="dataset for testing")
    parser.add_argument("--dataset_path", required=False, type=str, default=None, help="location for document-summary pairs, should be a csv file")
    parser.add_argument("--response_dir", required=True, type=str, help="location for generated responses, should be a directory")
    parser.add_argument("--behavior_response_save_dir", required=True, type=str, help="location for saving generated responses for evaluating behaviors, should be a directory")
    parser.add_argument("--behavior_results_save_path", required=True, type=str, help="location for saving evaluated behavior results, should be a json file")
    args = parser.parse_args()
    return args.marker_model_path, args.gpus, args.dataset_name, args.dataset_path, args.response_dir, args.behavior_response_save_dir, args.behavior_results_save_path


def get_LLMPredictor(model_path, tensor_parallel_size, sampling_params, save_dir):

    class LLMPredictor:

        def __init__(self):
            # Create an LLM.
            self.llm = LLM(model=model_path,
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


def get_create_prompt(behavior_prompt: str, response_dir: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def create_prompt(row: dict[str, Any]) -> dict[str, Any]:
        question_index = row["question_index"]
        with open(os.path.join(response_dir, f"{question_index}.txt")) as f:
            response = f.read().strip()
        row["response"] = response
        row["text"] = behavior_prompt.format(question=row["question"], correct_answer=row["correct_answer"], response=row["response"])
        return row
    return create_prompt


def main():

    marker_model_path, gpus, dataset_name, dataset_path, response_dir, behavior_response_save_dir, behavior_results_save_path = parse_args()
    assert (dataset_name is None or dataset_path is None) and (dataset_name is not None or dataset_path is not None), "Please provide either a dataset_name or a dataset_path"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    assert is_valid_cuda_visible_devices, "Invalid CUDA_VISIBLE_DEVICES value"

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=4096)
    # Set tensor parallelism per instance.
    tensor_parallel_size = 1
    # Set number of instances. Each instance will use tensor_parallel_size GPUs.
    num_instances = len(gpus.split(","))

    if dataset_path is not None:
        pass
    elif dataset_name is not None:
        if dataset_name == "medqa":
            dataset_path = os.path.join("data", "processed", "medqa_test.csv")
        elif dataset_name == "medmcqa":
            dataset_path = os.path.join("data", "processed", "medmcqa_dev.csv")
        else:
            raise ValueError(f"dataset {dataset_name} is not supported")
    else:
        raise ValueError("Please provide either a dataset_name or a dataset_path")
    df = pd.read_csv(dataset_path)

    # append response to df
    df["response"] = ""
    for i, row in df.iterrows():
        question_index = row["question_index"]
        with open(os.path.join(response_dir, f"{question_index}.txt")) as f:
            response = f.read().strip()
        df.at[i, "response"] = response

    count_match = r"<count>\s*(\d+)\s*</count>"
    average_behavior_count_dict = {}
    for behavior_name, behavior_prompt in zip(["verification_prompt", "backtracking_prompt", "subgoal_prompt", "backward_chaining_prompt"], 
                                              [verification_prompt, backtracking_prompt, subgoal_prompt, backward_chaining_prompt]):
        ds = ray.data.from_pandas(df)
        ds = ds.map(get_create_prompt(behavior_prompt, response_dir))
        ds = ds.repartition(128)

        resources_kwarg: dict[str, Any] = {}
        resources_kwarg["num_gpus"] = 1

        behavior_save_dir = os.path.join(behavior_response_save_dir, behavior_name)
        llm_predictor = get_LLMPredictor(marker_model_path, tensor_parallel_size, sampling_params, behavior_save_dir)
        os.makedirs(behavior_save_dir, exist_ok=True)

        ds = ds.map_batches(
            llm_predictor,
            concurrency=num_instances,
            batch_size=32,
            **resources_kwarg,
        )

        os.makedirs("temp_dir", exist_ok=True)
        ds.write_parquet("temp_dir")
        shutil.rmtree("temp_dir")

        # loop through save path and extract counts
        behavior_counts_list = []
        for file in os.listdir(behavior_save_dir):
            with open(os.path.join(behavior_save_dir, file)) as f:
                evaluation_response = f.read()
            try:
                behavior_counts = int(re.search(count_match, evaluation_response).group(1))
                behavior_counts_list.append(behavior_counts)
            except:
                pass

        average_behavior_count = sum(behavior_counts_list) / len(behavior_counts_list)
        average_behavior_count_dict[behavior_name] = average_behavior_count

    with open(behavior_results_save_path, "w") as f:
        json.dump(average_behavior_count_dict, f)


if __name__ == "__main__": 
    main()