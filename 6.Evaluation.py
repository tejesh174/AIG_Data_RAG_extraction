"""
06_evaluate.py
Step 6 - Merge extracted values with ground truth and compute accuracy metrics.

Output:
  results/results.csv
  results/evaluation_report.txt
"""

import sys
import math
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, RESULTS_DIR, GROUND_TRUTH_PATH, VARIABLES

EXTRACTED_PATH = RESULTS_DIR / "extracted.csv"
RESULTS_PATH = RESULTS_DIR / "results.csv"
REPORT_PATH = RESULTS_DIR / "evaluation_report.txt"

NUMERIC_TOLERANCE = 0.02


def normalize_category(value) -> str:
    """Normalize a categorical value for comparison."""
    text = str(value).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def segment_matches(ground_truth, extracted) -> bool:
    """Check if extracted categorical value matches ground truth.
    
    Uses flexible matching:
    1. Normalized substring match
    2. Check if all required segments from ground truth appear in extracted
    3. Check if key terms (general insurance, life and retirement) are present
    """
    if extracted is None or (isinstance(extracted, float) and math.isnan(extracted)):
        return False

    gt_str = normalize_category(ground_truth)
    ext_str = normalize_category(extracted)
    if not gt_str or not ext_str:
        return False

    # Direct normalized match
    if gt_str == ext_str:
        return True

    # Substring containment
    if gt_str in ext_str or ext_str in gt_str:
        return True

    # Check if all key segments from ground truth appear in extracted
    # Split ground truth on ' and ' to get individual segments
    gt_segments = [s.strip() for s in gt_str.split(" and ") if s.strip()]
    if gt_segments and all(seg in ext_str for seg in gt_segments):
        return True

    # Also handle semicolon-separated extracted values
    raw_segments = re.split(r"\s*[;,]\s*", str(extracted))
    ext_segments = [normalize_category(seg) for seg in raw_segments if normalize_category(seg)]
    if gt_segments and all(
        any(gt_seg in ext_seg or ext_seg in gt_seg for ext_seg in ext_segments)
        for gt_seg in gt_segments
    ):
        return True

    return False


def parse_number(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().replace(",", "").replace("$", "")
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    return float(text)


def is_match(var_type: str, ground_truth, extracted) -> bool:
    if var_type == "numeric":
        try:
            gt_val = parse_number(ground_truth)
            ext_val = parse_number(extracted)
            if gt_val is None or ext_val is None:
                return False
            if gt_val == 0:
                return ext_val == 0
            return abs((ext_val - gt_val) / gt_val) <= NUMERIC_TOLERANCE
        except (ValueError, TypeError):
            return False
    return segment_matches(ground_truth, extracted)


def numeric_error(ground_truth, extracted):
    try:
        gt_val = parse_number(ground_truth)
        ext_val = parse_number(extracted)
        if gt_val is None or ext_val is None:
            return None
        return round(ext_val - gt_val, 2)
    except Exception:
        return None


def pct_error(ground_truth, extracted):
    try:
        gt_val = parse_number(ground_truth)
        ext_val = parse_number(extracted)
        if gt_val in {None, 0} or ext_val is None:
            return None
        return round((ext_val - gt_val) / abs(gt_val) * 100, 2)
    except Exception:
        return None


def write_csv_with_fallback(df, path):
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        df.to_csv(fallback, index=False)
        print(f"[WARN] Could not write {path}; wrote {fallback} instead.")
        return fallback


def write_text_with_fallback(text, path):
    try:
        path.write_text(text)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        fallback.write_text(text)
        print(f"[WARN] Could not write {path}; wrote {fallback} instead.")
        return fallback


def run():
    print(f"\n{'='*60}")
    print("  Step 6 - Evaluation")
    print(f"{'='*60}\n")

    df_gt = pd.read_csv(GROUND_TRUTH_PATH)
    df_ext = pd.read_csv(EXTRACTED_PATH)

    df = df_gt.merge(
        df_ext[["company", "year", "variable", "extracted_value",
                "retrieved_pages", "num_chunks_used"]],
        on=["company", "year", "variable"],
        how="left",
    )

    var_type_map = {v["name"]: v["type"] for v in VARIABLES}
    df["var_type"] = df["variable"].map(var_type_map)

    df["match"] = df.apply(
        lambda r: is_match(r["var_type"], r["ground_truth"], r["extracted_value"]),
        axis=1,
    )
    df["absolute_error"] = df.apply(
        lambda r: numeric_error(r["ground_truth"], r["extracted_value"])
        if r["var_type"] == "numeric" else None,
        axis=1,
    )
    df["pct_error"] = df.apply(
        lambda r: pct_error(r["ground_truth"], r["extracted_value"])
        if r["var_type"] == "numeric" else None,
        axis=1,
    )

    col_order = [
        "company", "year", "variable", "var_type",
        "ground_truth", "extracted_value", "unit",
        "match", "absolute_error", "pct_error",
        "retrieved_pages", "num_chunks_used",
        "source_page_hint",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    results_path = write_csv_with_fallback(df, RESULTS_PATH)
    print(f"Results saved -> {results_path}\n")

    display_cols = ["company", "year", "variable", "ground_truth",
                    "extracted_value", "match", "pct_error"]
    print(df[display_cols].to_string(index=False))

    report_lines = []
    report_lines.append("=" * 62)
    report_lines.append("  EVALUATION REPORT")
    report_lines.append("=" * 62)

    overall_acc = df["match"].mean() * 100
    report_lines.append(f"\nOverall Accuracy:  {overall_acc:.1f}%  ({df['match'].sum()}/{len(df)} matches)\n")

    report_lines.append("Per-Variable Breakdown:")
    for var_name, grp in df.groupby("variable"):
        acc = grp["match"].mean() * 100
        vtype = grp["var_type"].iloc[0]
        line = f"  {var_name:<25} ({vtype:<11}) accuracy={acc:.0f}%"
        if vtype == "numeric":
            valid_pct = grp["pct_error"].dropna()
            if len(valid_pct) > 0:
                line += f"  | mean |pct_err|={valid_pct.abs().mean():.1f}%"
        report_lines.append(line)

    report_lines.append("\nRetrieval Stats:")
    report_lines.append(f"  Avg chunks used per extraction: {df['num_chunks_used'].mean():.1f}")
    report_lines.append(f"  Tolerance for numeric match:    {NUMERIC_TOLERANCE*100:.0f}%")

    report_lines.append("\nConfiguration:")
    from config import EMBEDDING_BACKEND, LLM_BACKEND, TOP_K_PAGES, CHUNK_SIZE
    report_lines.append(f"  Embedding backend : {EMBEDDING_BACKEND}")
    report_lines.append(f"  LLM backend       : {LLM_BACKEND}")
    report_lines.append(f"  Top-k pages       : {TOP_K_PAGES}")
    report_lines.append(f"  Chunk size        : {CHUNK_SIZE} chars")
    report_lines.append("=" * 62)

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_path = write_text_with_fallback(report_text, REPORT_PATH)
    print(f"\nReport saved -> {report_path}\n")


if __name__ == "__main__":
    run()
