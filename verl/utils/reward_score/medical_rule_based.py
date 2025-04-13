import os
import re

from random import random
from typing import Optional


def calculate_accuracy_reward(solution_str, gt_choice):
    if solution_str.count("\\boxed") == 1:
        match = re.search(r"\\boxed\{(.*?)\}", solution_str)
        if match:
            answer = match.group(1)
            if answer in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]:
                accuracy_reward = 1.0 if answer == gt_choice else 0.0
            else:
                accuracy_reward = -0.5
        else:
            accuracy_reward = -1.0
    else:
        accuracy_reward = -solution_str.count("\\boxed")
    return accuracy_reward


def calculate_english_percentage(solution_str):
    # Define English characters (A-Z, a-z, and common punctuation marks)
    english_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?;:'\"()[]{}<>-_/\\|`~@#$%^&*+=~0123456789 \n")
    # Remove non-English characters
    english_text = ''.join(char for char in solution_str if char in english_chars)
    # Calculate the percentage of English characters
    if len(solution_str) == 0:
        return 0.0
    else:
        return len(english_text) / len(solution_str)


def answer_with_summary_and_answer_box(s):
    # Ensure the pattern is not the only part of the string and no text comes after it
    s = s.strip()
    matches = re.findall(r'(\\boxed\{.*?\})', s)
    if len(matches) == 1:
        match = matches[0]
        return match != s
    else:
        return False


def calculate_format_reward(solution_str):
    # Define the pattern to match '<think>.*?</think>\s*<answer>.*?</answer>'
    pattern = r"^<think>.*?</think>\s*<answer>(.*?)</answer>$"

    # Check if the solution matches the pattern
    match = re.search(pattern, solution_str, re.DOTALL)
    if match:
        format_reward = 1.0
        if answer_with_summary_and_answer_box(match.group(1)):
            format_reward += 0.5
    else:
        format_reward = 0.0

    return format_reward


def compute_score(solution_str: str, ground_truth: str, add_bo_think_token: bool, logging_path: Optional[str], eos_token: str, step: int):
    gt_choice, choices = ground_truth.split("<SEP>", 1)
    question_index, choices = choices.split("<SEP>", 1)
    assert gt_choice in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    choices = [c.strip() for c in choices.split("<SEP>")]

    # handling undesirable text
    solution_str = solution_str.split(eos_token)[0].strip()
    solution_str = solution_str.replace("<\\think>", "</think>")
    solution_str = solution_str.replace("<\\answer>", "</answer>")
    # this is purely for FreedomIntelligence/HuatuoGPT-o1-8B response since it wont follow instructions
    if "## Final Response" in solution_str:
        solution_str = solution_str.split("## Final Response")[1].strip()
    # deepseek assistant chat template has <think> included
    if add_bo_think_token:
        solution_str = "<think>" + solution_str

    accuracy_reward = calculate_accuracy_reward(solution_str, gt_choice)
    format_reward = calculate_format_reward(solution_str)
    english_reward = calculate_english_percentage(solution_str)

    # logging
    if logging_path is not None:
        os.makedirs(logging_path, exist_ok=True)
        logging = solution_str + "\n\n-------------------------------------\n\n" + ground_truth + "\n\n-------------------------------------\n\n" + str(step) + "\n\n-------------------------------------\n\n" + "accuracy_reward: " + str(accuracy_reward) + "/1.0\n" + "format_reward: " + str(format_reward) + "/1.5\n" + "english_reward: " + str(english_reward) + "/1.0\n"
        with open(os.path.join(logging_path, f"{step}_{question_index}_{random()}.txt"), "w") as f:
            f.write(logging)

    if "train" in question_index:
        return accuracy_reward * 10 + format_reward + english_reward * 0.5
    elif "dev" in question_index:
        return accuracy_reward
    else:
        raise ValueError("Invalid question index")
