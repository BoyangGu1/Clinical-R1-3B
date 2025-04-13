import argparse
import os

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


response_template = """<think>
{reasoning}
</think>
<answer>
{response}
</answer>"""

system_prompt = "You are a helpful assistant. Answer the following multiple choice question. Your final answer MUST be included in the box \\boxed{}. For example, \\boxed{A} or \\boxed{B}."


def parse_args() -> tuple[str, str, int, str, int, int, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases_csv", required=False, default=os.path.join("data", "processed", "medqa_train_sft.csv"), type=str, help="The csv file that stores the case information. Should contain column question.")
    parser.add_argument("--save_dir", required=False, default=os.path.join("data", "distil", "medqa", "deepseek-r1"), type=str, help="The directory for saving distil responses")
    parser.add_argument("--total_api_no", required=False, default=1, type=int, help="The number of apis invovled for this round of generation.")
    parser.add_argument("--api", required=True, type=str, help="The api key for openai model generation.")
    parser.add_argument("--api_index", required=False, default="1",type=int, help="The api index in all apis.")
    parser.add_argument("--api_rate_limit", required=False, default="4",type=int, help="The api rate limit for each api.")
    parser.add_argument("--api_call_instance_index", required=True, type=int, help="Each api can make 4 calls at the same time. api_call_instance_index indicates which call is this one.")
    args = parser.parse_args()
    return args.cases_csv, args.save_dir, args.total_api_no, args.api, args.api_index, args.api_rate_limit, args.api_call_instance_index


def get_completion(client, prompt, system_prompt, model="deepseek-reasoner"):
    
    model_names = ["deepseek-chat", "deepseek-reasoner"]
    if model not in model_names:
        raise KeyError("model name not supported, try deepseek-chat or deepseek-reasoner")
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
        stream=False
    )

    return response.choices[0].message.content, response.choices[0].message.reasoning_content


def main():
    cases_csv, save_dir, total_api_no, api, api_index, api_rate_limit, api_call_instance_index = parse_args()

    # create a client for openai api
    client = OpenAI(
        api_key=api,
        base_url="https://api.deepseek.com",
    )

    os.makedirs(save_dir, exist_ok=True)
    # Split the cases so that each api instance will handle about same number of cases
    cases_df = pd.read_csv(cases_csv)
    total_cases = len(cases_df)
    # Select the cases that this specific api should handle
    cases_per_api = total_cases // total_api_no
    start_idx = (api_index - 1) * cases_per_api  # (api_index-1) because api_index starts from 1
    end_idx = start_idx + cases_per_api
    if api_index == total_api_no:
        end_idx = total_cases
    # Get the subset of cases for this api
    api_cases_df = cases_df.iloc[start_idx:end_idx]
    # Further split this for parallel calls
    cases_per_instance = len(api_cases_df) // api_rate_limit  # 4 parallel calls
    instance_start_idx = (api_call_instance_index - 1) * cases_per_instance
    instance_end_idx = instance_start_idx + cases_per_instance
    if api_call_instance_index == api_rate_limit:
        instance_end_idx = len(api_cases_df)
    # Get the final set of cases for this specific call instance
    api_instance_cases_df = api_cases_df.iloc[instance_start_idx:instance_end_idx]
    print(f"api {api_index} - Call instance {api_call_instance_index} will handle {len(api_instance_cases_df)} cases.")

    for i, row in tqdm(api_instance_cases_df.iterrows(), total=len(api_instance_cases_df)):
        if not os.path.exists(os.path.join(save_dir, f"{row['question_index']}.txt")):
            prompt = "Question: \n" + row["question"]
            if "Your final answer MUST be included in the box \\boxed{}. For example, \\boxed{A} or \\boxed{B}." in prompt:
                prompt = prompt.replace("Your final answer MUST be included in the box \\boxed{}. For example, \\boxed{A} or \\boxed{B}.", "")
            prompt = prompt.strip()
            # Generate the response
            try:
                response, reasoning = get_completion(client, prompt, system_prompt)
                with open(os.path.join(save_dir, f"{row['question_index']}.txt"), "w") as f:
                    f.write(response_template.format(reasoning=reasoning, response=response))
            except:
                pass


if __name__ == '__main__':
    main()