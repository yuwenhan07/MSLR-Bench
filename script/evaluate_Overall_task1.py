"""
Evaluation Task1: Field Score, Semantic Score and Overall Score
This script evaluates structured information extraction results for insider trading legal cases.

It compares predicted JSON outputs against gold-standard annotations on a field-by-field basis using
both exact match and semantic similarity (via embedding models). It computes accuracy metrics
(Field-Level Accuracy, Semantic Accuracy, and Overall Score) and outputs detailed and summary
reports for model performance analysis.
"""
import json
import os
import re
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import argparse

def normalize_amount(val):
    '''Normalize monetary value to float with two decimals'''
    if val is None:
        return None
    try:
        return str(round(float(re.sub(r'[^\d.]', '', val)), 2))
    except:
        return val

def normalize_date(val):
    '''Normalize date to YYYY-MM-DD format'''
    if val is None:
        return None
    val = re.sub(r'[^\d]', '-', val)
    parts = val.strip('-').split('-')
    if len(parts) == 3:
        return f"{parts[0]:0>4}-{parts[1]:0>2}-{parts[2]:0>2}"
    return val

# Check if two numeric values are close within a tolerance
def is_close(val1, val2, tolerance=0.05):
    try:
        v1, v2 = float(val1), float(val2)
        return abs(v1 - v2) / max(abs(v1), 1e-5) <= tolerance
    except:
        return False


STRICT_FIELDS = [
    "内幕交易信息的认定.内幕交易的股票名称",
    "内幕交易信息的认定.内幕信息形成时间",
    "内幕交易信息的认定.内幕信息公开时间",
    "当事人信息.当事人基础信息.姓名",
    "当事人信息.当事人基础信息.性别",
    "当事人信息.当事人基础信息.出生年份",
    "当事人信息.当事人基础信息.职务",
    "当事人信息.当事人的内幕交易认定.当事人知悉内幕交易时间",
    "当事人信息.当事人的内幕交易认定.买入/卖出",
    "当事人信息.当事人的内幕交易认定.买入时间",
    "当事人信息.当事人的内幕交易认定.买入金额（元）（最早买入时间均价）",
    "当事人信息.当事人的内幕交易认定.最早买入时间",
    "当事人信息.当事人的内幕交易认定.最晚买入时间",
    "当事人信息.当事人的内幕交易认定.基准日金额（元）",
    "当事人信息.当事人的内幕交易认定.违法所得（元）"
]

SEMANTIC_FIELDS = [
    "内幕交易信息的认定.内幕信息内容",
    "内幕交易信息的认定.内幕交易形成时间发生事项",
    "内幕交易信息的认定.内幕信息公开时间发生事项",
    "当事人信息.当事人的内幕交易认定.当事人角色",
]

# Extract value from nested dictionary using dot-separated path
def extract_field(data, field_path):
    for key in field_path.split("."):
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return str(data).strip() if data else None

# Extract numeric index from filename
def extract_index(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else None

# Compute exact match accuracy for structured fields
def field_accuracy(gold, pred, field_list):
    correct = 0
    total = 0
    for field in field_list:
        gold_val = extract_field(gold, field)
        pred_val = extract_field(pred, field)
        if gold_val is None and pred_val is None:
            correct += 1
            total += 1
            continue
        if (gold_val is not None and pred_val is None):
            total += 1
            continue
        if gold_val and pred_val:
            total += 1
            gold_val_norm = normalize_amount(gold_val) or normalize_date(gold_val) or gold_val
            pred_val_norm = normalize_amount(pred_val) or normalize_date(pred_val) or pred_val
            if gold_val_norm in pred_val_norm or pred_val_norm in gold_val_norm or is_close(gold_val_norm, pred_val_norm):
                correct += 1
    return correct / total if total else 0, total

# Compute semantic similarity score using embedding model
def semantic_similarity(gold, pred, field_list, semantic_model, threshold):
    correct = 0
    total = 0
    for field in field_list:
        gold_val = extract_field(gold, field)
        pred_val = extract_field(pred, field)
        if gold_val is None and pred_val is None:
            correct += 1
            total += 1
            continue
        if gold_val and pred_val:
            total += 1
            gold_val = gold_val.strip().lower()
            pred_val = pred_val.strip().lower()
            sim = util.cos_sim(
                semantic_model.encode(gold_val, convert_to_tensor=True),
                semantic_model.encode(pred_val, convert_to_tensor=True)
            ).item()
            if sim >= threshold:
                correct += 1
    return correct / total if total else 0, total

# Main evaluation function comparing gold and predicted outputs
def evaluate(gold_file, pred_file, embedding_model_path, threshold):
    semantic_model = SentenceTransformer(embedding_model_path)
    gold_data = []
    pred_data = []
    skipped = []

    gold_files_all = [f for f in os.listdir(gold_file) if f.endswith('.json')]
    pred_files_all = [f for f in os.listdir(pred_file) if f.endswith('.json')]
    
    pred_all_files = os.listdir(pred_file)
    txt_skipped_files = [f for f in pred_all_files if f.endswith('.txt')]
    for f in txt_skipped_files:
        skipped.append({"gold": None, "pred": f, "reason": "txt file skipped"})

    gold_dict = {int(re.search(r'\d+', f).group()): f for f in gold_files_all}
    pred_dict = {int(re.search(r'\d+', f).group()): f for f in pred_files_all}

    common_keys = sorted(set(gold_dict.keys()) & set(pred_dict.keys()))
    gold_files = [gold_dict[k] for k in common_keys]
    pred_files = [pred_dict[k] for k in common_keys]

    results = []

    for gf, pf in zip(gold_files, pred_files):
        if not gf.endswith('.json') or not pf.endswith('.json'):
            skipped.append({"gold": gf, "pred": pf, "reason": "non-json file"})
            print(f"Skipping non-json file pair: {gf}, {pf}")
            continue
        with open(os.path.join(gold_file, gf), 'r', encoding='utf-8') as fg, \
             open(os.path.join(pred_file, pf), 'r', encoding='utf-8') as fp:
            try:
                gd = json.load(fg)
                pd = json.load(fp)
                gold_data.append(gd)
                pred_data.append(pd)
            except Exception as e:
                skipped.append({"gold": gf, "pred": pf, "reason": str(e)})
                print(f"Error loading {gf} or {pf}: {e}")
                continue

    if isinstance(gold_data, dict): gold_data = [gold_data]
    if isinstance(pred_data, dict): pred_data = [pred_data]

    field_accs, sem_accs = [], []
    valid_gold_pred_pairs = []
    for gold, pred in tqdm(zip(gold_data, pred_data), total=len(gold_data)):
        valid_gold_pred_pairs.append((gold, pred))
    
    for gold, pred in tqdm(valid_gold_pred_pairs):
        field_acc, field_total = field_accuracy(gold, pred, STRICT_FIELDS)
        sem_acc, sem_total = semantic_similarity(gold, pred, SEMANTIC_FIELDS, semantic_model, threshold)
        field_accs.append(field_acc)
        sem_accs.append(sem_acc)

    print(f"Matched {len(valid_gold_pred_pairs)} JSON file pairs after skipping.")
    ALPHA = field_total / (field_total + sem_total) if field_total + sem_total > 0 else 0
    avg_field_acc = sum(field_accs) / len(field_accs)
    avg_sem_acc = sum(sem_accs) / len(sem_accs)
    overall_score = ALPHA * avg_field_acc + (1 - ALPHA) * avg_sem_acc

    results = []
    for i, (fa, sa) in enumerate(zip(field_accs, sem_accs)):
        results.append({
            "index": extract_index(gold_files[i]),
            "FieldAcc": fa,
            "SemAcc": sa,
            "OverallScore": ALPHA * fa + (1 - ALPHA) * sa
        })
    with open("eval_summary_2.json", "w", encoding="utf-8") as f:
        json.dump({
            "average": {
                "Field-Level Accuracy": avg_field_acc,
                "Semantic Accuracy": avg_sem_acc,
                "Overall Score": overall_score
            },
            "details": results,
            "metadata": {
                "total_files": len(gold_files),
                "matched_files": len(valid_gold_pred_pairs),
                "skipped_files": len(gold_files) - len(valid_gold_pred_pairs)
            },
        }, f, ensure_ascii=False, indent=2)

    with open("eval_skipped.json", "w", encoding="utf-8") as f:
        json.dump(skipped, f, ensure_ascii=False, indent=2)

    print(f"Field-Level Accuracy: {avg_field_acc:.4f}")
    print(f"Semantic Accuracy:    {avg_sem_acc:.4f}")
    print(f"Overall Score:        {overall_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_file', type=str, default="../data/processed", help='Path to gold standard directory')
    parser.add_argument('--pred_file', type=str, default="../output/task1", help='Path to predicted output directory')
    parser.add_argument('--embedding_model', type=str, default="../embedding_model/ChatLaw-Text2Vec", help='Path to embedding model')
    parser.add_argument('--semantic_threshold', type=float, default=0.6, help='Semantic similarity threshold')
    args = parser.parse_args()
    evaluate(args.gold_file, args.pred_file, args.embedding_model, args.semantic_threshold)
