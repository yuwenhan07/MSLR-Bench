"""
Evaluation Task1: FRC Score
This script evaluates the Field Completeness Rate (FCR) for a set of structured legal predictions.

It compares predicted JSON outputs against gold-standard annotations, checking how completely required
legal fields have been filled. The script supports flexible directory inputs and handles both strict 
and semantically inferred fields. It computes a completeness metric that reflects both structural validity 
and field-wise coverage.
"""
import os
import json
import re
import argparse

# Important Match field list
MATCH_FIELDS = [
    "内幕交易信息的认定.内幕信息形成时间",
    "内幕交易信息的认定.内幕交易的股票名称",
    "内幕交易信息的认定.内幕信息公开时间",
    "当事人信息.当事人基础信息.姓名",
    "当事人信息.当事人基础信息.性别",
    "当事人信息.当事人基础信息.出生年份",
    "当事人信息.当事人基础信息.职务",
    "当事人信息.当事人的内幕交易认定.当事人知悉内幕交易时间",
    "当事人信息.当事人的内幕交易认定.知悉方式类型",
    "当事人信息.当事人的内幕交易认定.买入/卖出",
    "当事人信息.当事人的内幕交易认定.买入时间",
    "当事人信息.当事人的内幕交易认定.买入金额（元）（最早买入时间均价）",
    "当事人信息.当事人的内幕交易认定.最早买入时间",
    "当事人信息.当事人的内幕交易认定.最晚买入时间",
    "当事人信息.当事人的内幕交易认定.基准日金额（元）",
    "当事人信息.当事人的内幕交易认定.违法所得（元）",
    "内幕交易信息的认定.内幕信息内容",
    "内幕交易信息的认定.内幕交易信息认定条款",
    "内幕交易信息的认定.内幕交易形成时间发生事项",
    "内幕交易信息的认定.内幕信息公开时间发生事项",
    "当事人信息.当事人的内幕交易认定.当事人角色",
    "当事人信息.当事人的内幕交易认定.当事人所属类型",
]


# Extract nested fields
def extract_field(data, field_path):
    """
    Extract the value of the specified field from a nested dictionary 'data' based on 'field_path'.
    The 'field_path' is dot-separated, indicating multi-level nested keys.
    If any key in the path does not exist or 'data' is not a dictionary, return None.
    The return value is a string with surrounding whitespace removed; if the value is None, return None.
    """
    for key in field_path.split('.'):
        if not isinstance(data, dict) or key not in data:
            return None  
        data = data[key] 
    return str(data).strip() if data is not None else None

def count_nonempty_fields(directory, match_fields):
    """
    Count the number of non-empty strict and semantic fields in all JSON files within the specified directory.
    Returns two values: a dictionary counting the number of files with non-empty fields for each field, and the total number of processed files.
    """
    match_counts = {field: 0 for field in match_fields}
    total_files = 0
    for filename in os.listdir(directory):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(directory, filename)
        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f) 
                total_files += 1 
                for field in match_fields:
                    val = extract_field(data, field)
                    if val:  
                        match_counts[field] += 1
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    print(f"✅ Processing completed: {total_files} JSON files")
    return match_counts, total_files

def compute_fcr_from_data(gold_data, pred_data, match_fields, N, M):
    """
    Calculate the Field Completeness Rate (FCR) based on ground truth data 'gold_data' and predicted data 'pred_data'.
    N is the total number of predicted files, M is the number of successfully read valid JSON files.
    For each sample, count the number of non-empty fields in the ground truth and corresponding non-empty fields in the prediction,
    then compute the average fill rate.
    The final FCR = (M/N) * (average field fill rate across all samples).
    """
    total_fields = match_fields  
    fcr_inner_sum = 0  
    i = 0 
    for gd, pd in zip(gold_data, pred_data):
        if not isinstance(pd, dict): 
            i +=1
            continue
        try:
            i += 1
            filled = 0  
            valid_fields = 0
            for field in total_fields:
                gold_val = extract_field(gd, field)
                if gold_val:  
                    valid_fields += 1
                    pred_val = extract_field(pd, field)
                    if pred_val:  
                        filled += 1
            if valid_fields > 0:
                fcr_inner_sum += filled / valid_fields  
        except Exception as e:
            print(f"Error comparing sample: {e}")  
            continue

    if N == 0 or M == 0:
        return 0.0  
    fcr_score = (M / N) * (fcr_inner_sum / M) 
    return fcr_score

def evaluate_fcr(gold_dir, pred_dir):
    """
    Compare JSON files in the ground truth directory 'gold_dir' and prediction directory 'pred_dir',
    and calculate the Field Completeness Rate (FCR).
    During processing, skip non-JSON files and output errors for failed file reads.
    """
    gold_data = []  
    pred_data = []  
    skipped = []  

    N = len([f for f in os.listdir(pred_dir) if f.endswith('.json') or f.endswith('.txt')])
    M = 0

    gold_files_all = [f for f in os.listdir(gold_dir) if f.endswith('.json')]  
    pred_files_all = [f for f in os.listdir(pred_dir) if f.endswith('.json')]  

    pred_all_files = os.listdir(pred_dir)
    txt_skipped_files = [f for f in pred_all_files if f.endswith('.txt')]  
    for f in txt_skipped_files:
        skipped.append({"gold": None, "pred": f, "reason": "txt file skipped"})  

    gold_dict = {int(re.search(r'\d+', f).group()): f for f in gold_files_all if re.search(r'\d+', f)}
    pred_dict = {int(re.search(r'\d+', f).group()): f for f in pred_files_all if re.search(r'\d+', f)}

    common_keys = sorted(set(gold_dict.keys()) & set(pred_dict.keys())) 
    gold_files = [gold_dict[k] for k in common_keys]
    pred_files = [pred_dict[k] for k in common_keys]

    for gf, pf in zip(gold_files, pred_files):
        with open(os.path.join(gold_dir, gf), 'r', encoding='utf-8') as fg, \
             open(os.path.join(pred_dir, pf), 'r', encoding='utf-8') as fp:
            try:
                gd = json.load(fg) 
                pd = json.load(fp) 
                gold_data.append(gd)
                pred_data.append(pd)
                M += 1  
            except Exception as e:
                skipped.append({"gold": gf, "pred": pf, "reason": str(e)})  
                print(f"Error loading {gf} or {pf}: {e}")
                continue

    fcr_score = compute_fcr_from_data(gold_data, pred_data, MATCH_FIELDS, N, M)
    print(f"✅ Successfully read valid JSON files: {M}")
    print(f"\n📈 Field Completeness Rate (FCR): {fcr_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Field Completeness Rate (FCR) for JSON structured data.")
    parser.add_argument('--data_dir', type=str, default="../output/task1", help='Directory containing predicted JSON files.')
    parser.add_argument('--gold_dir', type=str, default='../data/processed', help='Directory containing gold standard JSON files.')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"❌ Directory does not exist: {args.data_dir}")
        exit()

    match_counts, total_files = count_nonempty_fields(args.data_dir, MATCH_FIELDS)

    evaluate_fcr(args.gold_dir, args.data_dir)