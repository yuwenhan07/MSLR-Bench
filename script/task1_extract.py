"""
Task1 Structure Information Extraction
This script performs structured information extraction from Chinese legal case descriptions,
specifically focused on insider trading cases.

It loads a pretrained language model and iteratively processes each case to extract a fixed set of fields
into a standardized JSON structure. The model is prompted using a system/user message template and
the output is saved per case in either JSON or text format depending on parsing success.
"""
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../model/GLM4', type=str, help='PATH to the model')
    parser.add_argument('--data_path', default='../data/input_data.json', type=str, help='PATH to the dataset')
    parser.add_argument('--output_dir', default='../output/task1', type=str, help='Dirctory to save outputs')
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

    if not isinstance(data, list) or len(data) == 0:
        print("⚠️ The input data format is invalid or empty. Terminating processing.")
        return

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for item in tqdm(data[START_INDEX:], desc="processing"):
        case_id = item.get("序号", "unknown")
        description = item.get("案件描述", "")
        
        if not description.strip():
            continue

        prompt = build_prompt(description)

        messages = [
            {"role": "system", "content": "你是一个中文法律结构化抽取助手"},
            {"role": "user", "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(messages,
                                           add_generation_prompt=True,
                                           tokenize=True,
                                           return_tensors="pt",
                                           return_dict=True
                                           )
        inputs = inputs.to(device)
        
        # setting the generation kwargs
        gen_kwargs = {
            "max_length": 2048, 
            "do_sample": True, 
            "top_p": 0.9, 
            "temperature": 0.2
        }

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # 清洗 markdown 和控制字符
            output_text = output_text.replace("```json", "").replace("```", "").replace("\u200b", "").strip()
            import re
            match = re.search(r"\{[\s\S]*\}", output_text)
            if match:
                output_text = match.group(0)

        try:
            output_json = json.loads(output_text)
            Path(f"{OUTPUT_DIR}/output_{case_id}.json").write_text(json.dumps(output_json, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"✅ Case ID {case_id} has been saved as output_{case_id}.json")
        except json.JSONDecodeError:
            Path(f"{OUTPUT_DIR}/output_{case_id}.txt").write_text(output_text, encoding="utf-8")
            print(f"⚠️ Case ID {case_id} could not be parsed as JSON and was saved as output_{case_id}.txt")

def build_prompt(description):
    return f"""你是一位擅长处理法律文书和信息抽取任务的大型语言模型，特别擅长识别与内幕交易相关的法律事实。请你阅读以下案件描述，并从中提取指定字段，输出标准结构化 JSON。
【任务说明】
1. 请严格按照下方“字段提取范围”中列出的字段进行抽取；
2. 如果某些字段在描述中未出现，请保留该字段为空字符串 `""`；
3. 输出必须为合法的 JSON，字段名称及嵌套结构必须与“输出格式模板”完全一致；
4. 不需要生成说明文字或评论，仅输出 JSON 数据。
[案件描述]：{description}

【输出格式示例】
```json
{{
  "内幕交易信息的认定": {{
    "内幕交易的股票名称": "...",
    "内幕信息内容": "...",
    "内幕交易信息认定条款": "...",
    "内幕信息形成时间": "...",
    "内幕信息形成时间发生事项": "...",
    "内幕信息公开时间": "...",
    "内幕信息公开时间发生事项": "..."
  }},
  "当事人信息": {{
    "当事人基础信息": {{
      "姓名": "...",
      "性别": "...",
      "出生年份": "...",
      "文化程度": "",
      "职务": "..."
    }},
    "当事人的内幕交易认定": {{
      "当事人角色": "...",
      "当事人所属类型": "...",
      "当事人知悉内幕信息时间": "...",
      "知悉方式类型": "...",
      "当事人内幕交易所属类型": "...",
      "买入/卖出": "",
      "买入时间": "",
      "买入金额（元）（最早买入时间均价）": "",
      "最早买入时间": "",
      "最晚买入时间": "",
      "基准日金额（元）": "",
      "违法所得（元）": "..."
    }}
  }}
}}
""" 

if __name__ == "__main__":
    main()
