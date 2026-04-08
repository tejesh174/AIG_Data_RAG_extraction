"""
Central configuration for the AIG RAG Pipeline.

"""

import os
from pathlib import Path

os.environ["JAVA_HOME"] = r"C:\Users\tejes\AppData\Local\Programs\Eclipse Adoptium\jdk-17.0.18.8-hotspot"
os.environ["PYSPARK_PYTHON"] = r"C:\Users\tejes\AppData\Local\Programs\Python\Python39\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\tejes\AppData\Local\Programs\Python\Python39\python.exe"

# paths
ROOT = Path(r"C:\Users\tejes\aig-rag-project")
DATA_DIR        = ROOT / "data"
RAW_DIR         = DATA_DIR / "raw"
CHUNKS_DIR      = DATA_DIR / "chunks"
EMBEDDINGS_DIR  = DATA_DIR / "embeddings"
RESULTS_DIR     = ROOT / "results"

for d in [RAW_DIR, CHUNKS_DIR, EMBEDDINGS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.csv"
RESULTS_PATH      = RESULTS_DIR / "results.csv"

# AIG Company
COMPANY_NAME = "AIG"        
FILING_TYPE  = "10-K"

# Pulling 5 years of data
YEARS = [2017, 2018, 2019, 2020, 2021]

# extracting variables
# extracting variables
VARIABLES = [
    {
        "name": "total_revenues",
        "type": "numeric",
        "display": "Total Revenues",
        "unit": "millions USD",
        "extraction_hint": "Sum of all revenue line items in the Consolidated Statements of Income",
        "search_queries": [
            "total revenues",
            "Consolidated Statements of Income total revenues",
            "Consolidated Results of Operations total revenues",
        ],
    },
    {
        "name": "net_income",
        "type": "numeric",
        "display": "Net Income (Loss) Attributable to AIG",
        "unit": "millions USD",
        "extraction_hint": "Final net income line before per-share data in the Consolidated Statements of Income",
        "search_queries": [
            "Net income (loss) attributable to AIG",
            "Consolidated Statements of Income (Loss)",
            "Consolidated Results of Operations net income",
        ],
    },
    {
        "name": "industry",
        "type": "categorical",
        "display": "Primary Business Segments",
        "unit": None,
        "extraction_hint": "Core reportable segments described in Item 1 Business or Operating Structure",
        "search_queries": [
            "AIG operating structure core businesses General Insurance Life and Retirement",
            "business segments General Insurance Life and Retirement",
            "diversified mix of businesses reportable segments",
        ],
    },
]

# chunking
CHUNK_SIZE     = 1500
CHUNK_OVERLAP  = 400  
TOP_K_PAGES    = 5   

# Embedding models
EMBEDDING_BACKEND = "openai"
LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2" 
OPENAI_EMBED_MODEL = "text-embedding-3-small"

#LLM Extraction
# using open ai - gpt 4
LLM_BACKEND     = os.getenv("LLM_BACKEND", "openai")
OPENAI_LLM_MODEL = "gpt-4o"
#my secret open ai key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Pyspark
SPARK_APP_NAME = "AIG_RAG_Pipeline" 
SPARK_MASTER  = "local[*]"   #running on my machine using all cpu cores
