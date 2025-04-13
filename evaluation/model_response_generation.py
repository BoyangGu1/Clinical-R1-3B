import argparse
import os
import re
import shutil
from typing import Any, Callable, Optional

import ray
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams


thinking_user_prompt = """You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. Write a response that answers the following question. 
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

regular_user_prompt = """You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. Write a response that answers the following question. 

Question:
{question}
"""

thinking_cot_user_prompt = """You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. Write a response that answers the following question. 
Your response MUST and ONLY contain two parts: thinking and answer. Thinking part provides the reasoning of your response and MUST start with <think> and end with </think> tags. Answer part is the summary of the reasoning with the final answer and MUST start with <answer> and end with </answer> tags. Your reponse MUST contain <think>, </think>, <answer>, and </answer> tags. 
YOUR RESPONSE MUST STRICTLY BE OF THE FOLLOWING FORMAT:
<think>
Your thinking here.
</think>
<answer>
Summary of thinking and final answer here.
</answer>
- Verify that you have reached the answer and backtrack to the start or an intermediate step.
- Work backwards from the goal if it makes things easier.
- Decompose the answer into sub-goals and try to reach them to then reach the target, if you are unable to reach the goal or a subgoal backtrack to a previous state.

Question:
{question}
Your final choice MUST be included in the box \\boxed{{}}. For example, \\boxed{{A}} or \\boxed{{B}}."""


def parse_args() -> tuple[str, str, Optional[str], Optional[str], str, str, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str, help="the name of the model to test")
    parser.add_argument("--gpus", required=True, type=str, help="the gpu indices for this script to run on")
    parser.add_argument("--dataset_name", required=False, type=str, choices=["medqa", "medmcqa"], default=None, help="dataset name for testing")
    parser.add_argument("--dataset_path", required=False, type=str, default=None, help="location for document-summary pairs, should be a csv file")
    parser.add_argument("--save_dir", required=True, type=str, help="location for saving generated responses, should be a directory")
    parser.add_argument("--user_prompt_type", required=False, default="thinking_user_prompt", choices=["thinking_user_prompt", "regular_user_prompt", "thinking_cot_user_prompt"], help="use the thinking prompt instead of the regular prompt")
    parser.add_argument("--require_boxed_choice", action="store_true", help="whether over 98%% of response need to be contain a boxed choice")
    args = parser.parse_args()
    assert isinstance(args.require_boxed_choice, bool)
    return args.model_path, args.gpus, args.dataset_name, args.dataset_path, args.save_dir, args.user_prompt_type, args.require_boxed_choice


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


def get_create_prompt(user_prompt: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def create_prompt(row: dict[str, Any]) -> dict[str, Any]:
        row["text"] = user_prompt.format(question=row["question"])
        return row
    return create_prompt


def valid_response(response):
    pattern = r"\\boxed{(.*?)}"
    model_answers = re.findall(pattern, response)
    return model_answers and len(model_answers) == 1 and model_answers[0].strip().upper() in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


def get_non_valid_df_and_valid_response_ratio(full_df, save_dir):
    valid_question_indices_list = []
    for i, row in full_df.iterrows():
        question_index = row["question_index"]
        if os.path.exists(os.path.join(save_dir, f"{question_index}.txt")):
            with open(os.path.join(save_dir, f"{question_index}.txt"), "r") as f:
                response = f.read().strip()

            if valid_response(response):
                valid_question_indices_list.append(question_index)

    non_valid_df = full_df[~full_df["question_index"].isin(valid_question_indices_list)]
    valid_response_ratio = len(valid_question_indices_list) / len(full_df)

    return non_valid_df, valid_response_ratio


def main():

    model_path, gpus, dataset_name, dataset_path, save_dir, user_prompt_type, require_boxed_choice = parse_args()
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

    if require_boxed_choice:
        non_valid_df, valid_response_ratio = get_non_valid_df_and_valid_response_ratio(df, save_dir)
        valid_response_ratio_list = [valid_response_ratio]

        while not (valid_response_ratio > 0.98 or (len(valid_response_ratio_list) > 5 and len(set(valid_response_ratio_list[-5:])) == 1)):
            ds = ray.data.from_pandas(non_valid_df)
            assert user_prompt_type in ["thinking_user_prompt", "regular_user_prompt", "thinking_cot_user_prompt"]
            user_prompt = eval(user_prompt_type)
            ds = ds.map(get_create_prompt(user_prompt))
            ds = ds.repartition(128)

            resources_kwarg: dict[str, Any] = {}
            resources_kwarg["num_gpus"] = 1

            llm_predictor = get_LLMPredictor(model_path, tensor_parallel_size, sampling_params, save_dir)
            os.makedirs(save_dir, exist_ok=True)

            ds = ds.map_batches(
                llm_predictor,
                concurrency=num_instances,
                batch_size=32,
                **resources_kwarg,
            )

            os.makedirs("temp_dir", exist_ok=True)
            ds.write_parquet("temp_dir")
            shutil.rmtree("temp_dir")

            non_valid_df, valid_response_ratio = get_non_valid_df_and_valid_response_ratio(df, save_dir)
            valid_response_ratio_list.append(valid_response_ratio)

    else:
        ds = ray.data.from_pandas(df)
        assert user_prompt_type in ["thinking_user_prompt", "regular_user_prompt", "thinking_cot_user_prompt"]
        user_prompt = eval(user_prompt_type)
        ds = ds.map(get_create_prompt(user_prompt))
        ds = ds.repartition(128)

        resources_kwarg: dict[str, Any] = {}
        resources_kwarg["num_gpus"] = 1

        llm_predictor = get_LLMPredictor(model_path, tensor_parallel_size, sampling_params, save_dir)
        os.makedirs(save_dir, exist_ok=True)

        ds = ds.map_batches(
            llm_predictor,
            concurrency=num_instances,
            batch_size=32,
            **resources_kwarg,
        )

        os.makedirs("temp_dir", exist_ok=True)
        ds.write_parquet("temp_dir")
        shutil.rmtree("temp_dir")


if __name__ == "__main__": 
    main()