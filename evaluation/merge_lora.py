import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> tuple[str, str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True, type=str, help="model path for lora base model")
    parser.add_argument("--lora_model_path", required=True, type=str, help="model path for lora model")
    parser.add_argument("--merged_model_save_path", required=True, type=str, help="path to save the merged full-parameter model")
    args = parser.parse_args()
    return args.base_model_path, args.lora_model_path, args.merged_model_save_path


def main():
    base_model_path, lora_model_path, merged_model_save_path = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="cuda")
    lora_model = PeftModel.from_pretrained(base_model, lora_model_path)
    merged_model = lora_model.merge_and_unload()

    merged_model.save_pretrained(merged_model_save_path)
    tokenizer.save_pretrained(merged_model_save_path)


if __name__ == "__main__":
    main()