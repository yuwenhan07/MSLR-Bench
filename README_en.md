# MSLR: Multi-Step-Reasoning-Trace Chinese Multi-Step Legal Reasoning Benchmark Dataset

With the rapid development of large language models (LLMs) in legal applications, systematically evaluating their **reasoning ability in judgment prediction** has become increasingly urgent. Currently, publicly available legal evaluation benchmarks lack a unified evaluation framework and do not adequately support these two tasks. To fill this gap, we propose **MSLR**, addressing a critical gap in structured reasoning evaluation in the field of Chinese legal natural language processing, and providing a solid foundation for the evaluation and optimization of legal vertical domain large model systems. For more details, please refer to our paper.

## 📄 Introduction
MSLR is carefully designed to precisely evaluate large models' **legal document understanding and case analysis reasoning abilities**. We designed a semi-automated dataset construction scheme, building a comprehensive insider trading dataset through a **human + LLM** approach, which can also easily scale the dataset size and case types. On this basis, we designed two tasks: **structured information extraction** and **case fact analysis and judgment prediction**. To better evaluate model performance on these two tasks, we designed detailed and diverse evaluation methods, with **precise and comprehensive** results. Meanwhile, we created a **human-experience-based CoT (Chain-of-Thought) reasoning chain** designed by legal experts, aiming to test whether the model's reasoning ability improves when provided with reasoning chains, and whether it aligns more closely with human judicial processes.

## 📖 Data Sources and Construction Process

### Data Sources
The MSLR data mainly comes from the following three public channels, focusing on **insider trading cases** from 2005 to 2024, covering administrative cases, criminal cases, and non-prosecution cases:

- Administrative penalty decisions issued by the China Securities Regulatory Commission (CSRC);
- Criminal judgment documents published on the China Judgments Online;
- Non-prosecution decisions publicly disclosed by national procuratorates.

Each original legal document was collected in `.docx`, `.pdf`, or `.html` format, then converted into structured text for unified processing, while recording key information such as **document number, judgment date, and source link**. We removed some publicly available documents without specific case descriptions and outcomes, retaining only valid documents.

### Structured Data Design
In collaboration with legal experts, we developed a unified structured field system, detailed in `data/schema.json`, covering six major dimensions: insider information identification, basic party information, trading behavior, illegal gains calculation, applicable legal provisions, and final penalty results. All fields strictly align with the "Securities Law of the People's Republic of China," "Criminal Law of the People's Republic of China," "Civil Code," and other relevant statutes, ensuring **legal consistency, semantic comparability**, and structured support for judicial reasoning.

### Data Annotation Process
#### Manual Annotation
In the first phase, we manually annotated over **1000 real case documents** item by item, labeling more than **50,000 fields** according to the structured template. All annotations were completed by professionals with legal backgrounds, with **cross-review and sampling quality control mechanisms** in place to ensure consistency and accuracy of legal interpretation.
#### LLM-Assisted Extraction + Manual Verification
After initially establishing a high-quality seed set, we expansively processed over **400 additional case documents (2020–2024)**. After experimental comparison of different large model extraction strategies, we finally selected DeepSeek-V3, which offers a good balance of performance and cost-effectiveness, as the automatic extraction tool. To enhance its adaptability to legal tasks, we optimized the extraction prompt templates. All automatic outputs were manually verified by legal experts on a field-by-field basis to ensure structural completeness and semantic accuracy.

![data contruction](figs/data-contruction.png)

### Data Statistics Overview

| Metric                     | Value   |
|----------------------------|---------|
| Time Span                  | 2005–2024 |
| Total Number of Cases      | 1389    |
| Average Number of Fields per Document | 43.03   |
| Total Structured Field Entries | 59,771  |
| Average Core Field Completion Rate | 86.58%  |
| Average Number of Characters per Document | 2515.99 |

### Data Format
All raw and structured data are stored in the `data` folder. Processed data are saved in JSON file format, located in `data/processed`, which can be loaded using `json.load`. The `input.json` file summarizes the input parts from each JSON file for convenient usage.

## 🧩 Benchmark Task Definitions

### Task 1: LLM Automatic Annotation

This task aims to extract standardized key fields from legal case paragraphs, simulating the element summarization process performed by legal professionals after reading documents. Fields include insider information identification, party information, trading behavior, illegal gains, applicable legal provisions, and penalty results. Specific field definitions can be found in `data/extract_schema.json`.

- Input: Case description paragraph from the original legal document (natural language text)
- Output: Structured JSON format, filling in various field information
- Evaluation Metrics:
  - Field Accuracy: Strict matching accuracy of field values
  - Semantic Accuracy: Matching rate based on semantic similarity of fields
  - Overall Accuracy: Weighted composite score of the above two
  - Field Completeness Rate (FCR): Coverage and format completeness of output fields

### Task 2: Chinese Multi-step Legal Reasoning under the IRAC Framework
This task focuses on whether the model can generate logically rigorous and structurally complete legal analysis processes and final judgments based on case descriptions, evaluating reasoning quality and coverage of legal elements.

- Input Modes:
  - Standard Input (Std): Provide only case description, allowing the model to autonomously complete analysis and judgment
  - Chain-of-Thought Input (CoT): Provide case description + structured reasoning prompts, guiding the model to reason in the order of "Fact Identification → Legal Application → Judgment Result"
- Output Form: Natural language complete analysis process, covering core facts, applicable legal provisions, and judgment conclusions
- Evaluation Metrics:
  - LLM Score: Scores assigned by high-performance large language models based on logic, completeness, and legality (graded: A/B/C)
  - IRAC Recall: Measures consistency between model output and manually annotated structured fields (field-level matching)

#### CoT Reasoning Template Construction
Professional legal practitioners designed a Chain-of-Thought (CoT) prompt template based on human experience to guide the model in stepwise reasoning similar to judicial practice in insider trading cases. During design, full reference was made to the "Criminal Law of the People's Republic of China," "Civil Code," "Securities Law," and other relevant statutes, ensuring **rigor and authority** in legal application and expression. Furthermore, legal practitioners with years of trial experience participated in template construction, abstracting the common reasoning paths judicial authorities follow when handling insider trading cases based on **real adjudication logic**.
```text
Insider Information Formation → Information Awareness → Trading Behavior → Illegal Gains → Legal Application → Penalty Decision
```

## 📊 Experimental Design
In the experiments, we evaluated the performance of three major categories of models on the two tasks. Additionally, we provide example outputs from different models in the `example/` directory, visually demonstrating differences and capability boundaries of various models in real legal text processing.

### Experimental Models
- **General-purpose LLMs:** such as GPT-4, Qwen2.5, GLM4, DeepSeek-V3, etc., with comprehensive text understanding and generation capabilities;
- **Legal-domain LLMs:** such as CleverLaw, Lawyer-LLM, etc., fine-tuned on legal corpora, with stronger professionalism;
- **Reasoning-augmented LLMs:** such as DeepSeek-R1, QwQ-32B, etc., incorporating fast and slow thinking mechanisms, possessing stronger reasoning abilities.

## 🔧 How to Evaluate Models
This project provides a high-quality, structured Chinese legal judgment benchmark covering dual tasks of "structured information extraction" and "legal fact analysis and judgment prediction," and tests the impact of Chain-of-Thought prompts on model reasoning effectiveness.

### Environment Preparation
1. **Python Version:** Recommended Python ≥ 3.10
2. **Install Dependencies**
```bash
pip install -r requirements.txt
```
3. **Prepare Model Files:**
    - Place the large language model files or configurations to be evaluated in the `model/` directory
    - Download the Chinese legal semantic embedding model `ChatLaw-Text2Vec` and place it under the `embedding_model/` path for semantic similarity calculation

## Task Execution Process
### Run Chinese Multi-step Legal Reasoning based on the IRAC Framework
```bash
python script/predict.py \
  --model_path /path/to/model \
  --data_path ./data/input_data.json \
  --output_dir ./output
```
> During execution, both Std output and CoT output will be saved together in the output JSON file. To modify CoT prompts, directly edit the prompt in the Python script.

### Model Evaluation Scripts
#### Evaluate Task 1 LLM Automatic Annotation
**1. Overall Score (Field Accuracy + Semantic Similarity)**
```bash
python script/evaluate_Overall_task1.py \
  --gold_file data/processed \
  --pred_file output/task1 \
  --embedding_model embedding_model/ChatLaw-Text2Vec \
  --semantic_threshold 0.6
```

**2. FRC Score (Field Completeness Rate)**
```bash
python script/evaluate_FRC_task1.py \
  --data_dir ./output/task1 \
  --gold_dir ./data/processed
```
#### Evaluate Task 2 Chinese Multi-step Legal Reasoning under IRAC Framework
**1. LLM Score (Reasoning Quality Grade A/B/C, reviewed by model)**
```bash
python script/evaluate_LLMScore_task2.py \
  --gold_dir data/processed \
  --pred_dir output/task2 \
  --eval_scores_path result/llm_score_eval.json
```

**2. Relative Score (Consistency between reasoning output and structured fields)**
```bash
python script/evaluate_RelScore_task2.py \
  --gold_dir data/processed \
  --pred_dir output/task2 \
  --embedding_model embedding_model/ChatLaw-Text2Vec \
  --threshold 0.6 \
  --output_path result/relscore_task2.json
```

## 📎 Citation

If you use MSLR data or code, please cite our paper:
> The paper is currently under review; the repository will be updated after publication.

## 🛡️ Disclaimer

All legal data in MSLR are sourced from public channels, strictly anonymized, used only for research purposes, and strictly prohibited for real legal judgments.