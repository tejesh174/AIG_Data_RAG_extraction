"""
Evaluation: Merge extracted values with ground truth and compute evaluation metrics.

"""

import sys, math, re
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    RESULTS_DIR, GROUND_TRUTH_PATH, VARIABLES,
    EMBEDDING_BACKEND, OPENAI_LLM_MODEL, TOP_K_CHUNKS, CHUNK_SIZE,
)

EXTRACTED_PATH = RESULTS_DIR / "extracted.csv"
RESULTS_PATH   = RESULTS_DIR / "results.csv"
REPORT_PATH    = RESULTS_DIR / "evaluation_report.txt"

NUMERIC_TOLERANCE = 0.02


# Parsing & Matching

def parse_number(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().replace(",", "").replace("$", "")
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    return float(text)


def numeric_match(ground_truth, extracted):
    gt = parse_number(ground_truth)
    ext = parse_number(extracted)
    if gt is None or ext is None:
        return False
    if gt == 0:
        return ext == 0
    return abs((ext - gt) / gt) <= NUMERIC_TOLERANCE


def categorical_match(ground_truth, extracted):
    if extracted is None or (isinstance(extracted, float) and math.isnan(extracted)):
        return False
    gt = re.sub(r'[^a-z0-9]+', ' ', str(ground_truth).lower()).strip()
    ext = re.sub(r'[^a-z0-9]+', ' ', str(extracted).lower()).strip()
    if not gt or not ext:
        return False
    return gt == ext or gt in ext or ext in gt


def is_match(var_type, ground_truth, extracted):
    if var_type == "numeric":
        return numeric_match(ground_truth, extracted)
    return categorical_match(ground_truth, extracted)


def pct_error(ground_truth, extracted):
    gt = parse_number(ground_truth)
    ext = parse_number(extracted)
    if gt in {None, 0} or ext is None:
        return None
    return round((ext - gt) / abs(gt) * 100, 2)


def is_null(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if str(value).strip().lower() in {"", "nan", "none", "null"}:
        return True
    return False


# Main Function

def run():
    print(f"\n{'='*60}")
    print(f"  Step 6: Evaluation")
    print(f"{'='*60}\n")

    df_gt = pd.read_csv(GROUND_TRUTH_PATH)
    df_ext = pd.read_csv(EXTRACTED_PATH)

    # Merging on obs_id
    df = df_gt.merge(
        df_ext[["obs_id", "extracted_value", "num_chunks_used"]],
        on="obs_id", how="left",
    )

    # Variable types
    var_type_map = {v["name"]: v["type"] for v in VARIABLES}
    df["var_type"] = df["variable"].map(var_type_map)

    # Match, error, and null columns
    df["match"] = df.apply(
        lambda r: is_match(r["var_type"], r["ground_truth"], r["extracted_value"]), axis=1)
    df["pct_error"] = df.apply(
        lambda r: pct_error(r["ground_truth"], r["extracted_value"])
        if r["var_type"] == "numeric" else None, axis=1)
    df["is_null"] = df["extracted_value"].apply(is_null)

    # Saving results CSV
    df.to_csv(RESULTS_PATH, index=False)
    print(f"  Results saved -> {RESULTS_PATH}\n")

    # Printing results table
    print(df[["obs_id", "variable", "ground_truth", "extracted_value", "match", "pct_error", "is_null"]]
          .to_string(index=False))

    #Computing Metrics
    total = len(df)
    matches = df["match"].sum()
    nulls = df["is_null"].sum()
    non_nulls = total - nulls

    accuracy = matches / total * 100
    precision = (matches / non_nulls * 100) if non_nulls > 0 else 0
    recall = non_nulls / total * 100
    f1 = (2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
    null_rate = nulls / total * 100

    # Numeric-only metrics
    df_num = df[df["var_type"] == "numeric"].copy()
    df_num["gt_val"] = df_num["ground_truth"].apply(parse_number)
    df_num["ext_val"] = df_num["extracted_value"].apply(parse_number)
    df_num_valid = df_num.dropna(subset=["gt_val", "ext_val"])

    if len(df_num_valid) > 0:
        errors = df_num_valid["ext_val"] - df_num_valid["gt_val"]
        mae = errors.abs().mean()
        rmse = np.sqrt((errors ** 2).mean())
        mean_pct_err = df_num_valid["pct_error"].dropna().abs().mean()
    else:
        mae = rmse = mean_pct_err = None

    # Building Report
    lines = []
    lines.append("=" * 62)
    lines.append("  EVALUATION REPORT")
    lines.append("=" * 62)

    lines.append(f"\n  OVERALL METRICS")
    lines.append(f"  {'Accuracy:':<25} {accuracy:.1f}%  ({matches}/{total} matches)")
    lines.append(f"  {'Precision:':<25} {precision:.1f}%  ({matches}/{non_nulls} correct of non-null)")
    lines.append(f"  {'Recall:':<25} {recall:.1f}%  ({non_nulls}/{total} non-null extractions)")
    lines.append(f"  {'F1 Score:':<25} {f1:.1f}%")
    lines.append(f"  {'Null Rate:':<25} {null_rate:.1f}%  ({nulls}/{total} returned null)")

    if mae is not None:
        lines.append(f"\n  NUMERIC METRICS (on {len(df_num_valid)} valid extractions)")
        lines.append(f"  {'MAE:':<25} {mae:.1f} million USD")
        lines.append(f"  {'RMSE:':<25} {rmse:.1f} million USD")
        lines.append(f"  {'Mean |% Error|:':<25} {mean_pct_err:.2f}%")

    lines.append(f"\n  PER-VARIABLE BREAKDOWN")
    for var_name, grp in df.groupby("variable"):
        acc = grp["match"].mean() * 100
        vtype = grp["var_type"].iloc[0]
        n_null = grp["is_null"].sum()
        line = f"    {var_name:<30} accuracy={acc:.0f}%  nulls={n_null}/{len(grp)}"
        if vtype == "numeric":
            valid_pct = grp["pct_error"].dropna()
            if len(valid_pct) > 0:
                line += f"  mean|%err|={valid_pct.abs().mean():.2f}%"
        lines.append(line)

    lines.append(f"\n  RETRIEVAL STATS")
    lines.append(f"  {'Avg chunks used:':<25} {df['num_chunks_used'].mean():.1f}")
    lines.append(f"  {'Numeric tolerance:':<25} {NUMERIC_TOLERANCE*100:.0f}%")
    lines.append("=" * 62)

    report = "\n".join(lines)
    print("\n" + report)

    REPORT_PATH.write_text(report)
    print(f"\n  Report saved -> {REPORT_PATH}\n")


if __name__ == "__main__":
    run()