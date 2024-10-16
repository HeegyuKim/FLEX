# FLEX: False-Less EXecution for Text-to-SQL Evaluation
Paper: https://arxiv.org/pdf/2409.19014

## Introduction

FLEX (False-Less EXecution) is a novel approach to evaluating text-to-SQL systems, designed to overcome the limitations of existing metrics such as Execution Accuracy (EX). By leveraging Large Language Models (LLMs), FLEX emulates expert-level evaluation of SQL queries, providing a more accurate and nuanced assessment of text-to-SQL model performance.

Key features of FLEX include:

1. Comprehensive context analysis, considering natural language questions, database schemas, and external knowledge.
2. Sophisticated evaluation criteria with detailed guidelines for assessing query correctness.
3. Robust handling of noisy ground truth, correctly evaluating queries even when the provided ground truth is inaccurate or ambiguous.

Our evaluation shows that FLEX achieves significantly higher agreement with human expert judgments (Cohen's kappa of 87.04) compared to the existing EX metric (62.00) and outperforms previous LLM-based methods.

Using FLEX, we re-evaluated 50 publicly available text-to-SQL models on the Spider and BIRD benchmarks, revealing:

- Significant shifts in model rankings, particularly for the BIRD benchmark.
- Generally higher scores compared to EX, suggesting that FLEX captures previously underestimated aspects of model capability.
- Instances of overestimation in BIRD's challenging questions, highlighting areas for future research focus.

This repository provides the implementation of FLEX and tools for applying it to your own text-to-SQL evaluations.

## Setup

### 1. Install Required Packages

```bash
# Install PyTorch and Transformers for your environment
pip install torch

# Install other required packages
pip install -r requirements.txt

# Optional: Install additional frameworks if needed
pip install cohere together vllm
```

### 2. Download Datasets and Models

#### Spider Dataset
1. Download `spider_data.zip` from [Google Drive](https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view)
2. Unzip and move contents to the `spider` directory

#### BIRD Dataset
```bash
mkdir bird-download && cd bird-download
wget https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
unzip dev.zip # dev_20240627 directory will be created

# Move dev_20240627 directory to bird_dev/llm/data
mv dev_20240627/* ../bird_dev/llm/dev_databases/
```

### Directory Structure
Ensure your directory structure looks like this:
```
.
├── bird_dev/
│   └── llm/
│       └── data/
│           ├── dev_databases/
│           │   └── california_schools/
│           │       └── california_schools.sqlite
│           ├── dev.json
│           ├── dev_gold.sql
│           └── dev_tables.json
└── spider/
    ├── database/
    │   └── academic/
    │       ├── academic.sqlite
    │       └── academic.sql
    ├── dev_gold.sql
    ├── dev.json
    └── dev_tables.json
```

## Evaluation

### Spider FLEX Evaluation

```bash
# 1. Execute prediction file to get execution result
python -m spider_eval.evaluation --pred results/spider/SuperSQL.sql

# 2. Evaluate using FLEX
python llm_judge_main.py results/spider/SuperSQL_ex.json spider "gpt-4o-2024-08-06" --batch_size 32 --num_threads 8

# 3. Categorize error cases
python llm_judge_categorize.py "text2sql-baselines/spider/SuperSQL_ex_judge.jsonl" spider gpt-4o-2024-08-06 --batch_size 32 --num_threads 8
```

Note: GPT-4o-2024-08-06 costs approximately $4 per 1034 instances for judgment.

### BIRD FLEX Evaluation

```bash
# 1. Execute prediction file
bash bird_dev/eval.sh results/bird/SuperSQL.sql

# 2. Evaluate using FLEX
python llm_judge_main.py results/bird/SuperSQL_ex.json "bird-dev" "gpt-4o-2024-08-06" --batch_size 32 --num_threads 8

# 3. Categorize error cases
python llm_judge_categorize.py "results/bird/SuperSQL_ex_judge.jsonl" bird-dev gpt-4o-2024-08-06 --batch_size 32 --num_threads 8
```

Note: GPT-4o-2024-08-06 costs approximately $6 per 1534 instances for judgment.

## Human Agreement Evaluation

```bash
# Evaluate false positives
python llm_human_agreement.py results/agreement/fn.jsonl gpt-4o-2024-08-06

# Evaluate false negatives
python llm_human_agreement.py results/agreement/fp.jsonl gpt-4o-2024-08-06

# Calculate Cohen's Kappa score
python cohen_kappa_score.py results/agreement/fn_gpt-4o-2024-08-06.jsonl results/agreement/fp_gpt-4o-2024-08-06.jsonl
```

For more options, check [example scripts](./script/00-human-llm-agreement/) and [utils/llm_judge.py](./utils/llm_judge.py) for detailed implementation.