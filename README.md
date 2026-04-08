# AIG_Data_RAG_Pipeline
A retrieval-augmented generation pipeline that extracts financial data from AIG's 2025 10-K SEC filing.
Variables Extracted:
1. Net Premiums Written (numeric) — by segment and year
2. Underwriting Income (numeric) — by segment and year
3. Primary Regulatory Authority (categorical) — by jurisdiction
Each variable has 5 ground truth observations (15 total).

Pipeline Steps:
Step 1 - Data_Extraction
Downloads AIG 10-K from SEC EDGAR and converts HTML to plain text

Step 2 - Ground_Truth Dataset
Builds the ground truth dataset (15 observations manually verified from the filing)

Step 3 - Chunking and Embedding
Chunks the document using PySpark (2000 chars, 600 overlap) and generates OpenAI embeddings

Step 4 - Retrieval
Retrieves top-8 chunks per observation using cosine similarity + lexical boosting

Step 5 - Extraction
Extracts values from retrieved chunks using GPT-4o with structured JSON output

Step 6 - Evaluation
Merges extractions with ground truth and computes accuracy metrics

Results:
==============================================================
  EVALUATION REPORT
==============================================================
  OVERALL METRICS:
  Accuracy:                 93.3%  (14/15 matches)
  Precision:                93.3%  (14/15 correct of non-null)
  Recall:                   100.0%  (15/15 non-null extractions)
  F1 Score:                 96.6%
  Null Rate:                0.0%  (0/15 returned null)

SETUP:
pip install -r requirements.txt
Update config.py with your paths and OpenAI API key, then run scripts in order:
1.Data_Extraction.py
2.Ground_Truth.py
3.Chunk_Embed.py
4.Retrieval.py
5.Extraction.py
6.Evaluation.py

TECH STACK:
Python + PySpark
OpenAI text-embedding-3-small
GPT-4o Open AI

