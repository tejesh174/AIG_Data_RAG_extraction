"""
Extract target variables from retrieved chunks using an LLM.
"""

import sys
import json
import re
import time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    RESULTS_DIR, VARIABLES,
    LLM_BACKEND, OPENAI_LLM_MODEL,
    OPENAI_API_KEY, TOP_K_PAGES,
    COMPANY_NAME, YEARS,
)

RETRIEVED_PATH = RESULTS_DIR / "retrieved_chunks.csv"
EXTRACTED_PATH = RESULTS_DIR / "extracted.csv"
MAX_CONTEXT_CHUNKS = TOP_K_PAGES


SYSTEM_PROMPT = """You are a financial data extraction assistant.
You will be given text excerpts from a company's annual SEC filing.
Extract ONLY the specific value requested.

Rules:
- Financial tables often show multiple fiscal years side by side.
- Return the value ONLY for the specific fiscal year requested.
- Match the year in the column header to the correct data column.
- Negative values shown in parentheses should be returned as negative numbers.
- Return ONLY valid JSON with no explanation or markdown."""


def make_user_prompt(company: str, year: int, variable: dict, context: str) -> str:
    var_type = variable["type"]
    var_display = variable["display"]
    unit = variable.get("unit") or "N/A"
    hint = variable.get("extraction_hint", "")

    if var_type == "numeric":
        instructions = (
            f"Extract '{var_display}' for {company} for fiscal year {year}.\n"
            f"Unit: {unit}.\n"
        )
        if hint:
            instructions += f"Hint: {hint}\n"
        instructions += (
            "\nReturn the value as a number in millions, no commas or currency symbols.\n"
            "Parenthesized values are negative.\n"
            "If not found in the excerpts, return null.\n\n"
            'Return JSON: {"extracted_value": <number or null>}'
        )
    else:
        instructions = (
            f"Extract '{var_display}' for {company} as of fiscal year {year}.\n"
        )
        if hint:
            instructions += f"Hint: {hint}\n"
        instructions += (
            "\nReturn the value as a concise string based on what the filing states.\n"
            "If not found in the excerpts, return null.\n\n"
            'Return JSON: {"extracted_value": <string or null>}'
        )

    return (
        f"Company: {company}\nFiscal Year: {year}\n\n"
        f"--- DOCUMENT EXCERPTS ---\n{context}\n"
        f"--- END EXCERPTS ---\n\n"
        f"{instructions}"
    )

#calling openai
def call_openai(system: str, user: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=256,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


#routing to configured llm backend
def call_llm(system: str, user: str) -> str:
    if LLM_BACKEND == "openai":
        return call_openai(system, user)
    else:
        raise RuntimeError(
            f"Unsupported LLM_BACKEND='{LLM_BACKEND}'"
        )


#parsing llm response with extracted value and raw response
def parse_llm_response(raw: str) -> tuple:
    try:
        clean = re.sub(r"```json\s*|```", "", raw).strip()
        data = json.loads(clean)
        extracted = data.get("extracted_value")
        if isinstance(extracted, str) and extracted.strip().lower() in {"", "null", "none", "n/a"}:
            extracted = None
        return extracted, raw
    except Exception:
        return None, raw


def run():
    print(f"\n{'='*60}")
    print(f"  LLM Extraction")
    print(f"  Backend: {LLM_BACKEND.upper()} ({OPENAI_LLM_MODEL})")
    print(f"  Company: {COMPANY_NAME}  |  Years: {YEARS}")
    print(f"  Max context chunks per query: {MAX_CONTEXT_CHUNKS}")
    print(f"{'='*60}\n")

    if not OPENAI_API_KEY:
        sys.exit(
            "OPENAI_API_KEY is not set; skipping extraction so "
            "existing results/extracted.csv is not overwritten."
        )

    if not RETRIEVED_PATH.exists():
        sys.exit(
            f"Retrieved chunks not found: {RETRIEVED_PATH}\n"
        )

    df_ret = pd.read_csv(RETRIEVED_PATH)
    print(f"  Loaded {len(df_ret):,} retrieved chunks from {RETRIEVED_PATH.name}\n")

    results = []
    groups = df_ret.groupby(["company", "year", "variable"])
    total = len(groups)

    for i, ((company, year, var_name), grp) in enumerate(groups, 1):
        var_cfg = next((v for v in VARIABLES if v["name"] == var_name), None)
        if var_cfg is None:
            continue

        # Building context from top-scoring unique chunks
        grp_sorted = grp.sort_values("score", ascending=False).drop_duplicates("chunk_uid")
        context_parts = []
        for _, row in grp_sorted.head(MAX_CONTEXT_CHUNKS).iterrows():
            context_parts.append(
                f"[Source: {company} {year} 10-K | "
                f"Chunk {row['chunk_uid']} | page ~{row['page_est']}]\n"
                f"{row['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        user_prompt = make_user_prompt(company, int(year), var_cfg, context)

        print(f"  [{i}/{total}] {company} {year} - {var_cfg['display']} ...",
              end=" ", flush=True)

        try:
            raw_resp = call_llm(SYSTEM_PROMPT, user_prompt)
            extracted_value, raw = parse_llm_response(raw_resp)
            print(f"-> {extracted_value!r}")
        except Exception as e:
            extracted_value, raw = None, str(e)
            print(f"-> ERROR: {e}")

        results.append({
            "company":          company,
            "year":             year,
            "variable":         var_name,
            "extracted_value":  extracted_value,
            "retrieved_pages":  sorted(grp_sorted["page_est"].unique().tolist()),
            "num_chunks_used":  min(len(grp_sorted), MAX_CONTEXT_CHUNKS),
            "raw_llm_response": raw,
        })

        time.sleep(0.3)

    #saving all the results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_extracted = pd.DataFrame(results)
    df_extracted.to_csv(EXTRACTED_PATH, index=False)
    print(f"\n  Extracted {len(df_extracted)} values to {EXTRACTED_PATH}\n")

    #Summary
    print(df_extracted[["company", "year", "variable", "extracted_value"]]
          .to_string(index=False))
    print()


if __name__ == "__main__":
    run()