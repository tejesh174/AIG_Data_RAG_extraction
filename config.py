"""
Central configuration for the AIG RAG Pipeline.
AIG 2025 10-K from SEC EDGAR
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


# Company & Filing
COMPANY_NAME = "AIG"
FILING_TYPE  = "10-K"
FISCAL_YEAR  = 2025

# SEC EDGAR HTML source
FILING_HTML_URL = (
    "https://www.sec.gov/Archives/edgar/data/5272/"
    "000000527226000023/aig-20251231.htm"
)
# Variables
VARIABLES = [
    {
        "name": "net_premiums_written",
        "type": "numeric",
        "display": "Net Premiums Written",
        "unit": "millions USD",
        "extraction_hint": (
            "Net premiums written by segment from the Business Segment "
            "Consolidated Results of Operations"
        ),
        "search_queries": [
            "net premiums written",
            "underwriting results",
            "net premiums written segment operations years ended december",
            "business segment operations net premiums",
        ],
    },
    {
        "name": "underwriting_income",
        "type": "numeric",
        "display": "Underwriting Income",
        "unit": "millions USD",
        "extraction_hint": (
            "Underwriting income by segment from the Business Segment Operations "
            "Consolidated Results of Operations"
        ),
        "search_queries": [
            "underwriting income",
            "underwriting results",
            "underwriting income segment operations years ended december",
            "underwriting income loss combined ratio",
        ],
    },
    {
        "name": "regulatory_jurisdiction",
        "type": "categorical",
        "display": "Primary Regulatory Authority",
        "unit": None,
        "extraction_hint": (
            "The primary regulatory authority overseeing AIG's "
            "International "
            "insurance operations in each jurisdiction"
        ),
        "search_queries": [
            "regulatory authority insurance regulator",
            "lead regulator insurance operations",
            "insurance regulatory regime jurisdiction supervisor",
            "regulated by insurance commissioner authority",
        ],
    },
]

# Observations
OBSERVATIONS = {
    "net_premiums_written": [
        {"obs_id": "NPW_NAC_2025", "segment": "North America Commercial", "year": 2025},
        {"obs_id": "NPW_IC_2025",  "segment": "International Commercial",  "year": 2025},
        {"obs_id": "NPW_GP_2025",  "segment": "Global Personal",           "year": 2025},
        {"obs_id": "NPW_NAC_2024", "segment": "North America Commercial", "year": 2024},
        {"obs_id": "NPW_IC_2024",  "segment": "International Commercial",  "year": 2024},
    ],
    "underwriting_income": [
        {"obs_id": "UW_NAC_2025", "segment": "North America Commercial", "year": 2025},
        {"obs_id": "UW_IC_2025",  "segment": "International Commercial",  "year": 2025},
        {"obs_id": "UW_GP_2025",  "segment": "Global Personal",           "year": 2025},
        {"obs_id": "UW_NAC_2024", "segment": "North America Commercial", "year": 2024},
        {"obs_id": "UW_IC_2024",  "segment": "International Commercial",  "year": 2024},
    ],
    "regulatory_jurisdiction": [
        {"obs_id": "REG_US",        "jurisdiction": "United States"},
        {"obs_id": "REG_UK",        "jurisdiction": "United Kingdom"},
        {"obs_id": "REG_EU",        "jurisdiction": "European Union (Luxembourg)"},
        {"obs_id": "REG_SINGAPORE", "jurisdiction": "Singapore"},
        {"obs_id": "REG_JAPAN",     "jurisdiction": "Japan"},
    ],
}

# Chunking
CHUNK_SIZE     = 2000
CHUNK_OVERLAP  = 600
TOP_K_CHUNKS   = 8

# Embedding
EMBEDDING_BACKEND  = "openai"
LOCAL_EMBED_MODEL  = "all-MiniLM-L6-v2"
OPENAI_EMBED_MODEL = "text-embedding-3-small"

# LLM Extraction
LLM_BACKEND      = os.getenv("LLM_BACKEND", "openai")
OPENAI_LLM_MODEL = "gpt-4o"
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")


# PySpark
SPARK_APP_NAME = "AIG_RAG_Pipeline"
SPARK_MASTER   = "local[*]"