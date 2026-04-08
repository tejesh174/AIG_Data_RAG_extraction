"""
Building the ground truth dataset.
5 observations per variable (15 total)
All values verified from: https://www.sec.gov/Archives/edgar/data/5272/000000527226000023/aig-20251231.htm

"""

import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import GROUND_TRUTH_PATH, COMPANY_NAME, FISCAL_YEAR, VARIABLES

GROUND_TRUTH_RECORDS = [
    # ── Net Premiums Written (5 observations)
    # Source: MD&A - Business Segment Operations tables (pages 44-48)
    {"obs_id": "NPW_NAC_2025", "variable": "net_premiums_written",
     "segment": "North America Commercial", "year": 2025,
     "ground_truth": 8759, "unit": "millions USD",
     "source_section": "MD&A - North America Commercial Segment Operations"},

    {"obs_id": "NPW_IC_2025", "variable": "net_premiums_written",
     "segment": "International Commercial", "year": 2025,
     "ground_truth": 8663, "unit": "millions USD",
     "source_section": "MD&A - International Commercial Segment Operations"},

    {"obs_id": "NPW_GP_2025", "variable": "net_premiums_written",
     "segment": "Global Personal", "year": 2025,
     "ground_truth": 6253, "unit": "millions USD",
     "source_section": "MD&A - Global Personal Segment Operations"},

    {"obs_id": "NPW_NAC_2024", "variable": "net_premiums_written",
     "segment": "North America Commercial", "year": 2024,
     "ground_truth": 8452, "unit": "millions USD",
     "source_section": "MD&A - North America Commercial Segment Operations"},

    {"obs_id": "NPW_IC_2024", "variable": "net_premiums_written",
     "segment": "International Commercial", "year": 2024,
     "ground_truth": 8364, "unit": "millions USD",
     "source_section": "MD&A - International Commercial Segment Operations"},

    # Underwriting Income (5 observations)
    # Source: MD&A - Business Segment Operations tables (pages 44-48)
    {"obs_id": "UW_NAC_2025", "variable": "underwriting_income",
     "segment": "North America Commercial", "year": 2025,
     "ground_truth": 1144, "unit": "millions USD",
     "source_section": "MD&A - North America Commercial Segment Operations"},

    {"obs_id": "UW_IC_2025", "variable": "underwriting_income",
     "segment": "International Commercial", "year": 2025,
     "ground_truth": 1118, "unit": "millions USD",
     "source_section": "MD&A - International Commercial Segment Operations"},

    {"obs_id": "UW_GP_2025", "variable": "underwriting_income",
     "segment": "Global Personal", "year": 2025,
     "ground_truth": 70, "unit": "millions USD",
     "source_section": "MD&A - Global Personal Segment Operations"},

    {"obs_id": "UW_NAC_2024", "variable": "underwriting_income",
     "segment": "North America Commercial", "year": 2024,
     "ground_truth": 548, "unit": "millions USD",
     "source_section": "MD&A - North America Commercial Segment Operations"},

    {"obs_id": "UW_IC_2024", "variable": "underwriting_income",
     "segment": "International Commercial", "year": 2024,
     "ground_truth": 1227, "unit": "millions USD",
     "source_section": "MD&A - International Commercial Segment Operations"},

    # Regulatory Jurisdiction (5 observations)
    # Source: Item 1 Business - Regulation (pages 7-8)
    {"obs_id": "REG_US", "variable": "regulatory_jurisdiction",
     "segment": "United States", "year": 2025,
     "ground_truth": "NYDFS", "unit": None,
     "source_section": "Item 1 Business - Regulation - United States - States"},

    {"obs_id": "REG_UK", "variable": "regulatory_jurisdiction",
     "segment": "United Kingdom", "year": 2025,
     "ground_truth": "PRA", "unit": None,
     "source_section": "Item 1 Business - Regulation - International"},

    {"obs_id": "REG_EU", "variable": "regulatory_jurisdiction",
     "segment": "European Union (Luxembourg)", "year": 2025,
     "ground_truth": "Commissariat aux Assurances", "unit": None,
     "source_section": "Item 1 Business - Regulation - International"},

    {"obs_id": "REG_SINGAPORE", "variable": "regulatory_jurisdiction",
     "segment": "Singapore", "year": 2025,
     "ground_truth": "MAS", "unit": None,
     "source_section": "Item 1 Business - Regulation - International"},

    {"obs_id": "REG_JAPAN", "variable": "regulatory_jurisdiction",
     "segment": "Japan", "year": 2025,
     "ground_truth": "JFSA", "unit": None,
     "source_section": "Item 1 Business - Regulation - International"},
]

#ground truth function
def build_ground_truth():
    print(f"\n{'='*60}")
    print(f" Building Ground Truth Dataset")
    print(f" Company: {COMPANY_NAME} | Filing: {FISCAL_YEAR} 10-K")
    print(f" Variables: {[v['name'] for v in VARIABLES]}")
    print(f"{'='*60}\n")

    #ground truth dataframe
    df = pd.DataFrame(GROUND_TRUTH_RECORDS)
    df.insert(0, "company", COMPANY_NAME)

    # converting dataframe to csv
    df.to_csv(GROUND_TRUTH_PATH, index=False)
    print(f"\n  Ground truth saved to {GROUND_TRUTH_PATH}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    df = build_ground_truth()