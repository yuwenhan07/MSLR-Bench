"""
Evaluation Task2: Relative Score
This script evaluates the relative reasoning performance of legal judgment prediction models.

It compares predicted outputs against gold-standard structured data by computing how well each model
(standard vs. chain-of-thought) covers both strictly defined factual fields and semantically rich fields.

It supports semantic similarity scoring using a SentenceTransformer embedding model and outputs per-case
scores along with aggregate statistics, including average scores and skipped evaluations.
"""
import json
import os
import re
from tqdm import tqdm
import argparse
from sentence_transformers import SentenceTransformer, util

# List of fields requiring exact match
STRICT_FIELDS = [
    "内幕交易信息的认定.内幕信息形成时间",
    "内幕交易信息的认定.内幕交易的股票名称",
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
    "当事人信息.当事人的内幕交易认定.违法所得（元）",
    "当事人信息.当事人处罚结果.没收违法所得金额（元）",
    "当事人信息.当事人处罚结果.罚款倍数",
    "当事人信息.当事人处罚结果.罚款数额（元）"
]

# List of fields requiring exact match
SEMANTIC_FIELDS = [
    "内幕交易信息的认定.内幕信息内容", 
    "内幕交易信息的认定.内幕交易信息认定条款",
    "内幕交易信息的认定.内幕交易信息所属类型",
    "内幕交易信息的认定.内幕交易形成时间发生事项",
    "内幕交易信息的认定.内幕信息公开时间发生事项",
    "当事人信息.当事人的内幕交易认定.当事人角色",
    "当事人信息.当事人的内幕交易认定.当事人所属类型",
    "当事人信息.当事人的内幕交易认定.当事人知悉内幕信息的方式（原文）",
    "当事人信息.当事人的内幕交易认定.知悉方式类型",
    "当事人信息.当事人的内幕交易认定.当事人内幕交易所属类型",
    "当事人信息.当事人处罚结果.处罚依据",
]


def load_json(file_path):
    """
    Load JSON file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_field_values(data):
    """
    Extract field values used for strict and semantic matching from the sample
    """
    def normalize_value(val):
        return str(val).strip()

    IGNORED_KEYS = {
        "序号",
        "案件信息",
        "当事人信息.当事人处罚结果申辩",
        "法律文书原文",
        "案件描述",
        "案件分析",
        "最终判决"
    }

    strict_values = {}
    semantic_values = {}

    def walk(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_path = f"{path}.{k}" if path else k
                if any(new_path.startswith(key) for key in IGNORED_KEYS):
                    continue
                walk(v, new_path)
        elif isinstance(obj, list):
            for item in obj:
                walk(item, path)
        else:
            if obj is not None:
                norm_val = normalize_value(obj)
                if norm_val != "-":
                    if path in STRICT_FIELDS:
                        strict_values[path] = norm_val
                    elif path in SEMANTIC_FIELDS:
                        semantic_values[path] = norm_val

    walk(data)
    return strict_values, semantic_values

def semantic_match(prediction_text, field_values, model, threshold=0.60):
    """
    Use sentence embedding model for semantic matching between field values and prediction text
    """
    slices = [s.strip() for s in prediction_text.split('，') if s.strip()]
    slice_embeddings = model.encode(slices, convert_to_tensor=True)
    field_embeddings = model.encode(list(field_values), convert_to_tensor=True)

    matched = 0
    for emb in field_embeddings:
        cos_scores = util.cos_sim(emb, slice_embeddings)
        if cos_scores.max() > threshold:
            matched += 1
    return matched

def evaluate_prediction(entry, output, model, threshold=0.60):
    """
    Score and evaluate a single prediction result
    """
    strict_values, semantic_values = extract_field_values(entry)
    prediction_texts = {
        "std": output.get('std', ''),
        "cot": output.get('cot', '')
    }

    scores = {}
    for name, text in prediction_texts.items():
        if text:
            strict_matches = 0
            for field, val in strict_values.items():
                norm_val = str(val).strip()
                if norm_val in text:
                    strict_matches += 1
            semantic_matches = semantic_match(text, semantic_values.values(), model, threshold=threshold)

            total = len(strict_values) + len(semantic_values)
            final_score = (strict_matches + semantic_matches) / total if total > 0 else 0

            scores[name] = {
                "strict_matched": strict_matches,
                "semantic_matched": semantic_matches,
                "strict_total": len(strict_values),
                "semantic_total": len(semantic_values),
                "score": round(final_score, 4)
            }
    return scores


def evaluate(gold_file, pred_file, model_path, threshold=0.60, output_path="score_result.json"):
    """
    Batch evaluate all prediction results, calculate average scores and save
    """
    model = SentenceTransformer(model_path)
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
    std_scores = []
    cot_scores = []

    for gf, pf in tqdm(zip(gold_files, pred_files), total=len(gold_files), desc="Evaluating"):
        with open(os.path.join(gold_file, gf), 'r', encoding='utf-8') as fg, \
             open(os.path.join(pred_file, pf), 'r', encoding='utf-8') as fp:
            try:
                gd = json.load(fg)
                pd = json.load(fp)
                res = evaluate_prediction(gd, pd, model, threshold=threshold)
                results.append({
                    "file": pf,
                    "score_detail": res
                })
                if "std" in res:
                    std_scores.append(res["std"]["score"])
                if "cot" in res:
                    cot_scores.append(res["cot"]["score"])
            except Exception as e:
                skipped.append({"gold": gf, "pred": pf, "reason": str(e)})
                print(f"Error loading {gf} or {pf}: {e}")
                continue

    avg_std = round(sum(std_scores) / len(std_scores), 4) if std_scores else 0.0
    avg_cot = round(sum(cot_scores) / len(cot_scores), 4) if cot_scores else 0.0
    average_score = round((avg_std + avg_cot) / 2, 4)

    final_result = {
        "average_score": average_score,
        "average_std": avg_std,
        "average_cot": avg_cot,
        "details": results,
        "skipped": skipped
    }
    print(avg_std, avg_cot)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Command line interface: requires passing gold_dir, pred_dir and embedding_model
    parser = argparse.ArgumentParser(description="Evaluate prediction results for RelScore task 2.")
    parser.add_argument('--gold_dir', type=str, default="../data/processed", help='Directory containing gold standard JSON files.')
    parser.add_argument('--pred_dir', type=str, default="../output/task2", help='Directory containing prediction JSON files.')
    parser.add_argument('--embedding_model', type=str, default="../embedding_model/ChatLaw-Text2Vec", help='Path to the SentenceTransformer model.')
    parser.add_argument('--threshold', type=float, default=0.6, help='Semantic similarity threshold for matching.')
    parser.add_argument('--output_path', type=str, default="score_result.json", help='Path to save the evaluation results.')
    args = parser.parse_args()
    evaluate(args.gold_dir, args.pred_dir, args.embedding_model, threshold=args.threshold, output_path=args.output_path)