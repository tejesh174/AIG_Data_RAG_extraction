"""
Building the ground truth dataset.

"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import GROUND_TRUTH_PATH, COMPANY_NAME, VARIABLES, YEARS


GROUND_TRUTH_RECORDS = [
    # Total revenues.
    {"year": 2021, "variable": "total_revenues", "ground_truth": 52057,
     "unit": "millions USD", "source_page_hint": "Consolidated Statements of Income"},
    {"year": 2020, "variable": "total_revenues", "ground_truth": 43736,
     "unit": "millions USD", "source_page_hint": "Consolidated Statements of Income"},
    {"year": 2019, "variable": "total_revenues", "ground_truth": 49746,
     "unit": "millions USD", "source_page_hint": "Consolidated Statements of Income"},
    {"year": 2018, "variable": "total_revenues", "ground_truth": 47389,
     "unit": "millions USD", "source_page_hint": "Consolidated Statements of Income"},
    {"year": 2017, "variable": "total_revenues", "ground_truth": 49520,
     "unit": "millions USD", "source_page_hint": "Consolidated Statements of Income"},

    # Net income (loss) attributable to AIG.
    {"year": 2021, "variable": "net_income", "ground_truth": 9388,
     "unit": "millions USD", "source_page_hint": "Consolidated Statements of Comprehensive Income (Loss)"},
    {"year": 2020, "variable": "net_income", "ground_truth": -5944,
     "unit": "millions USD", "source_page_hint": "Consolidated Statements of Comprehensive Income (Loss)"},
    {"year": 2019, "variable": "net_income", "ground_truth": 3348,
     "unit": "millions USD", "source_page_hint": "Consolidated Statements of Comprehensive Income (Loss)"},
    {"year": 2018, "variable": "net_income", "ground_truth": -6,
     "unit": "millions USD", "source_page_hint": "Consolidated Statements of Comprehensive Income (Loss)"},
    {"year": 2017, "variable": "net_income", "ground_truth": -6084,
     "unit": "millions USD", "source_page_hint": "Consolidated Statements of Comprehensive Income (Loss)"},

    # Primary industry/segment.
    {"year": 2021, "variable": "industry", "ground_truth": "General Insurance and Life and Retirement",
     "unit": None, "source_page_hint": "ITEM 1 | Business"},
    {"year": 2020, "variable": "industry", "ground_truth": "General Insurance and Life and Retirement",
     "unit": None, "source_page_hint": "ITEM 1 | Business"},
    {"year": 2019, "variable": "industry", "ground_truth": "General Insurance and Life and Retirement",
     "unit": None, "source_page_hint": "ITEM 1 | Business"},
    {"year": 2018, "variable": "industry", "ground_truth": "General Insurance and Life and Retirement",
     "unit": None, "source_page_hint": "ITEM 1 | Business"},
    {"year": 2017, "variable": "industry", "ground_truth": "General Insurance and Life and Retirement",
     "unit": None, "source_page_hint": "ITEM 1 | Business"},
]


def build_ground_truth():
    print(f"\n{'='*60}")
    print(" Building Ground Truth Dataset")
    print(f" Company: {COMPANY_NAME} | Variables: {[v['name'] for v in VARIABLES]}")
    print(f"{'='*60}\n")

    df = pd.DataFrame(GROUND_TRUTH_RECORDS)
    df.insert(0, "company", COMPANY_NAME)

    counts = df.groupby("variable").size()
    print("  Observations per variable:")
    for var, count in counts.items():
        status = "OK" if count == 5 else "BAD"
        print(f"    {status} {var}: {count}")

    assert (counts == 5).all()

    df.to_csv(GROUND_TRUTH_PATH, index=False)
    print(f"\nGround truth saved -> {GROUND_TRUTH_PATH}")
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    df = build_ground_truth()
