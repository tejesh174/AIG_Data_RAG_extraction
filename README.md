# AIG_Data_RAG_Pipeline

A retrieval-augmented generation pipeline that extracts financial data from AIG's 2025 10-K SEC filing.

## Variables Extracted

1. **Net Premiums Written** (numeric) — by segment and year
2. **Underwriting Income** (numeric) — by segment and year
3. **Primary Regulatory Authority** (categorical) — by jurisdiction

Each variable has 5 ground truth observations (15 total).

## Pipeline Steps

**Step 1 - Data_Extraction**
Downloads AIG 10-K from US SEC Filings and extracts plain text

**Step 2 - Ground_Truth Dataset**
Builds the ground truth dataset (15 observations manually verified from the filing)

**Step 3 - Chunking and Embedding**
Chunks the document using PySpark (2000 chars, 600 overlap) and generates OpenAI embeddings

**Step 4 - Retrieval**
Retrieves top-8 chunks per observation using cosine similarity + lexical boosting

**Step 5 - Extraction**
Extracts values from retrieved chunks using GPT-4o with structured JSON output

**Step 6 - Evaluation**
Merges extractions with ground truth and computes accuracy metrics

## Results

```
==============================================================
  EVALUATION REPORT
==============================================================

  OVERALL METRICS:
  Accuracy:                 93.3%  (14/15 matches)
  Precision:                93.3%  (14/15 correct of non-null)
  Recall:                   100.0%  (15/15 non-null extractions)
  F1 Score:                 96.6%
  Null Rate:                0.0%  (0/15 returned null)
```

## Setup

```
pip install -r requirements.txt
```

Update config.py with your paths and OpenAI API key, then run scripts in order:

```
python 1_Data_Extraction.py
python 2_Ground_Truth.py
python 3_Chunk_Embed.py
python 4_Retrieval.py
python 5_Extraction.py
python 6_Evaluation.py
```

## Tech Stack

- Python + PySpark
- OpenAI text-embedding-3-small
- GPT-4o OpenAI

## AIG_PYSPARK — Full PySpark Implementation
The `AIG_PYSPARK/` folder contains an alternative version of the same pipeline that maximizes PySpark usage across all 6 steps, not just chunking. Both versions produce identical results (93.3% accuracy, 14/15 correct).

| Feature | AIG-RAG-FINAL (main) | AIG_PYSPARK |
|---------|---------------------|-------------|
| Spark usage | Chunking only | All 6 steps |
| Spark SQL | Minimal | Data validation, chunk distribution analysis, retrieval summaries |
| Window functions | — | Chunk ranking, error analysis |
| UDFs | — | Match evaluation, null detection |
| Caching | — | Multi-pass DataFrame operations |
| Best for | Single filing, fast iteration | Batch processing at scale |

Built both to compare trade-offs — the main version prioritizes simplicity and speed for single filings, while the PySpark version demonstrates how the pipeline would scale for processing hundreds of filings in parallel.
