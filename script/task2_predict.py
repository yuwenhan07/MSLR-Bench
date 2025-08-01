"""
Task2 Legal Fact Analysis and Judgement Prediction
This script performs dual-mode legal judgment prediction for insider trading cases using a pretrained language model.

It takes a dataset of legal case descriptions and generates two outputs for each case:
1. A standard reasoning-based judgment (std)
2. A chain-of-thought guided judgment (cot)

The results are saved in JSON format for each case, enabling further comparative evaluation of reasoning strategies.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from pathlib import Path
from tqdm import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../model/GLM4', type=str, help='Path to the model')
    parser.add_argument('--data_path', default='../data/input_data.json', type=str, help='Path to the dataset')
    parser.add_argument('--output_dir', default='../output/task2', type=str, help='Directory to save outputs')
    parser.add_argument('--start_index', default=0, type=int, help='Start index for processing')
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    DATA_PATH = args.data_path
    OUTPUT_DIR = args.output_dir
    START_INDEX = args.start_index

    # setting devices
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Loading tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
        ).eval()  

    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Fail to load Dataset: {e}")
        return

    for item in tqdm(data[START_INDEX:], desc="Processing"):
        case_id = item.get("序号", "unknown")
        description = item.get("案件描述", "")
        
        if not description.strip():
            continue

        prompts = {
            "std": build_prompt(description),
            "cot": build_prompt_cot(description),
        }

        gen_kwargs = {
            "max_length": 4096, 
            "do_sample": True, 
            "top_p": 0.9, 
            "temperature": 0.2
        }

        result = {}
        for label, selected_prompt in prompts.items():
            messages = [
                {"role": "system", "content": "你是一个中文法律案情分析与判决预测助手"},
                {"role": "user", "content": selected_prompt},
            ]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs, num_return_sequences=1)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                result[label] = output_text

        Path(f"{OUTPUT_DIR}/output_{case_id}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ Case ID {case_id} has been saved as output_{case_id}.json")

def build_prompt(description):
    return f"""你是一名法律判决生成专家，专注于内幕交易案件的裁判分析。请认真阅读下列案件事实，结合法律条文和证据材料，进行全面的法律分析，分析当事人行为，并最终生成完整、具有法律依据的裁决结论。
    [案件描述]：{description}"""

def build_prompt_cot(description):
    return f"""你是一名法律判决生成专家，专注于内幕交易案件的裁判分析。请认真阅读下列案件事实，结合法律条文和证据材料，按照**内幕信息形成 → 信息知悉 → 交易行为 → 违法所得 → 法律适用与判决类型 → 处罚结果**的推理链条，展开全面的法律分析，推导当事人行为的合法性，并最终生成完整、具有法律依据的裁决结论。

请确保推理过程清晰、逻辑严密，语言规范、术语专业，结论应具备法律说服力与可执行性。
    [案件描述]：{description}"""

if __name__ == "__main__":
    main()