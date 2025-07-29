# MSLR: Multi-Step-Reasoning-Trace - A Chinese Benchmark for Multi-step Legal Reasoning

With the rapid development of Large Language Models (LLMs) in the legal domain, it has become increasingly urgent to evaluate their **reasoning capabilities in legal judgment prediction**. Existing legal benchmarks lack a unified evaluation framework and provide limited support for structured reasoning tasks. To address this gap, we introduce **MSLR**, a benchmark that fills a critical void in the evaluation of structured legal reasoning in Chinese and provides a solid foundation for assessing and optimizing legal-domain LLMs. For more details, please refer to our paper.

## 📄 Introduction

MSLR is meticulously designed to evaluate LLMs' abilities in **legal document understanding and case reasoning**. We propose a semi-automated data construction pipeline combining **manual efforts and LLMs**, resulting in a comprehensive insider trading dataset. This approach supports easy expansion in both volume and case diversity. Based on this dataset, we define two core tasks: **Structured Information Extraction** and **Case Fact Analysis with Judgment Prediction**. We also develop diverse evaluation metrics to ensure **accurate and comprehensive assessment**.

To examine whether models benefit from guided reasoning, we design a **Chain-of-Thought (CoT) reasoning path** based on human legal reasoning, aiming to test whether such guidance aligns model behavior more closely with real-world judicial processes.

## 📖 Data Sources and Construction

### Data Sources

MSLR data is primarily collected from the following three public sources, focusing on insider trading cases between 2005 and 2024, covering various legal stages:

- Administrative penalty decisions from the China Securities Regulatory Commission (CSRC)
- Criminal judgments from China Judgments Online
- Non-prosecution decisions from national procuratorates

Each legal document was collected in `.docx`, `.pdf`, or `.html` format and converted into structured text. Metadata such as **document ID, decision date, and source link** was preserved. Documents without substantive case descriptions or outcomes were discarded.

### Structured Data Schema

In collaboration with legal professionals, we created a unified schema (see `data/schema.json`) covering six dimensions: insider information identification, party details, trading behavior, illicit gains, applicable laws, and penalty results. These fields are aligned with the *Securities Law*, *Criminal Law*, and *Civil Code* to ensure **legal consistency and structured reasoning support**.

### Annotation Process

#### Manual Annotation

In the first phase, we manually annotated over **1,000 real-world legal documents**, resulting in **over 50,000 field entries**. Legal experts performed annotations based on templates, supported by cross-validation and sampling checks to ensure consistency and accuracy.

#### LLM-Assisted Extraction with Human Verification

Building on the seed set, we expanded to **over 400 new cases (2020–2024)**. After evaluating multiple extraction strategies, we selected **DeepSeek-V3** for its balance of performance and cost. We optimized prompt templates for legal scenarios, and all outputs were **manually verified field-by-field** by legal experts to ensure structural and semantic correctness.

![data contruction](figs/data-contruction.png)

### Data Overview

| Metric | Value |
|--------|-------|
| Time Span | 2005–2024 |
| Number of Cases | 1,389 |
| Avg. Fields per Document | 43.03 |
| Total Structured Fields | 59,771 |
| Avg. Core Field Completion | 86.58% |
| Avg. Words per Document | 2515.99 |

### Data Format

All raw and processed data is stored in the `data` folder in JSON format. Processed data is under `data/processed`, and can be loaded with `json.load`. The `input.json` file consolidates the input sections for ease of use.

## 🧩 Benchmark Task Definition: Chinese Multi-step Legal Reasoning under the IRAC Framework

This task assesses whether models can generate logically consistent and structured legal analyses and judgments based on case descriptions.

- **Input Modes:**
  - **Standard (Std):** Only the case description is provided
  - **Chain-of-Thought (CoT):** Case description + structured reasoning hints guiding the model to reason in the sequence of "fact identification → legal application → judgment outcome"

- **Output:** Natural language analysis covering core facts, applicable laws, and final judgment

- **Evaluation Metrics:**
  - **LLM Score:** Graded (A/B/C) by a high-performing model based on logic, completeness, and legality
  - **IRAC Recall:** Field-level consistency with human-annotated structured fields

### CoT Reasoning Template

Legal professionals constructed a CoT template grounded in real-world judicial reasoning practices, especially for insider trading. The reasoning chain follows:

```text
Insider Info Formation → Awareness → Trading Behavior → Illicit Gains → Legal Application → Penalty Decision
```

## 📊 Experimental Results

We evaluated three types of models on both tasks. Example outputs from different models are available in the `example/` folder to illustrate practical performance differences and reasoning boundaries.

- **General-purpose LLMs:** GPT-4, Qwen2.5, GLM4, DeepSeek-V3, etc.
- **Legal-domain LLMs:** CleverLaw, Lawyer-LLM
- **Reasoning-augmented LLMs:** DeepSeek-R1, QwQ-32B

## 🔧 How to Evaluate Models

MSLR provides a high-quality, structured legal benchmark covering both **Information Extraction** and **Legal Reasoning & Judgment Prediction**, and evaluates CoT prompting effects.

### Environment Setup

1. Python ≥ 3.10
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Prepare models:
   - Place LLMs under `model/`
   - Download `ChatLaw-Text2Vec` for semantic similarity and put under `embedding_model/`

## 🧪 Running the Tasks

### Run Legal Reasoning Task (IRAC-based)
```bash
python script/predict.py \
  --model_path /path/to/model \
  --data_path ./data/input_data.json \
  --output_dir ./output
```

> Std and CoT outputs are saved together. Modify the CoT prompt directly in the script.

### Evaluation Scripts

**1. LLM Score**
```bash
python script/LLMScore.py \
  --gold_dir data/processed \
  --pred_dir ./output \
  --eval_scores_path result/llm_score_eval.json
```

**2. IRAC Recall**
```bash
python script/IRAC.py \
  --gold_dir data/processed \
  --pred_dir output/task2 \
  --embedding_model embedding_model/ChatLaw-Text2Vec \
  --threshold 0.6 \
  --output_path result/relscore_task2.json
```

## 📎 Citation

If you use MSLR data or code, please cite our paper (to be updated after acceptance):
> Paper under review. Will be added once available.

## 🛡️ Disclaimer

All legal data in MSLR is publicly sourced and anonymized. For research use only — not for real-world legal decision-making.