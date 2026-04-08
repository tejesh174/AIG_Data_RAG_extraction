"""
Extracting target variables from retrieved chunks using OpenAI LLM.

"""

import sys, json, re, time
from pathlib import Path
import pandas as pd
import openai

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    RESULTS_DIR, VARIABLES, OBSERVATIONS,
    OPENAI_LLM_MODEL, OPENAI_API_KEY,
    TOP_K_CHUNKS, COMPANY_NAME, FISCAL_YEAR,
)

RETRIEVED_PATH = RESULTS_DIR / "retrieved_chunks.csv"
EXTRACTED_PATH = RESULTS_DIR / "extracted.csv"

SYSTEM_PROMPT = """You are a financial data extraction assistant.
Extract ONLY the specific value requested from the given SEC 10-K filing excerpts.
Rules:
- Match the EXACT fiscal year requested. Tables often show multiple years side by side (e.g., 2025, 2024, 2023). Extract ONLY from the requested year column.
- Match the EXACT segment name requested. Do not confuse values across segments.
- Distinguish between similar line items: 'Net premiums written' and 'Net premiums earned' are DIFFERENT metrics. Only extract the one requested.
- Parenthesized values like (70) mean negative: return -70.
- If the requested value is not clearly present in the excerpts, return null rather than guessing.
- Return ONLY valid JSON, no explanation."""

#extracting the values using llm
def extract_value(obs, variable, context):
    segment = obs.get("segment", obs.get("jurisdiction", ""))
    year = obs.get("year", FISCAL_YEAR)
    vtype = variable["type"]

    if vtype == "numeric":
        ask = (f"Extract '{variable['display']}' for {COMPANY_NAME} '{segment}' "
               f"segment, fiscal year {year}. Unit: millions USD.\n"
               f"IMPORTANT: The table may show multiple years. Extract ONLY the value from the {year} column.\n"
               f"Do NOT confuse 'net premiums written' with 'net premiums earned'.\n"
               f"Use the EXACT number from the financial table, not rounded figures from narrative text.\n"
               'Return JSON: {{"extracted_value": <number or null>}}')
    else:
        ask = (f"Extract the primary regulatory authority for {COMPANY_NAME} "
               f"operations in {segment}. Use abbreviation if exists. "
               'Return JSON: {{"extracted_value": <string or null>}}')

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"--- EXCERPTS ---\n{context}\n--- END ---\n\n{ask}"},
        ],
        temperature=0, max_tokens=256,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content

    # Parsing JSON response
    try:
        val = json.loads(re.sub(r"```json\s*|```", "", raw).strip()).get("extracted_value")
        if isinstance(val, str) and val.strip().lower() in {"", "null", "none"}:
            val = None
        return val, raw
    except:
        return None, raw


def run():
    print(f"\n{'='*60}")
    print(f"  Step 5: LLM Extraction ({OPENAI_LLM_MODEL})")
    print(f"{'='*60}\n")

    if not OPENAI_API_KEY:
        sys.exit("OPENAI_API_KEY is not set.")
    if not RETRIEVED_PATH.exists():
        sys.exit("Retrieved chunks not found. Run Step 4 first.")

    df_ret = pd.read_csv(RETRIEVED_PATH)
    print(f"  Loaded {len(df_ret):,} retrieved chunks\n")

    results = []
    total = sum(len(v) for v in OBSERVATIONS.values())
    count = 0

    for var in VARIABLES:
        var_name = var["name"]
        for obs in OBSERVATIONS.get(var_name, []):
            count += 1
            obs_id = obs["obs_id"]
            segment = obs.get("segment", obs.get("jurisdiction", ""))

            # Building context from top retrieved chunks
            grp = (df_ret[(df_ret["variable"] == var_name) & (df_ret["obs_id"] == obs_id)]
                   .sort_values("score", ascending=False).head(TOP_K_CHUNKS))
            context = "\n\n---\n\n".join(
                f"[Chunk {r['chunk_uid']} | page ~{r['page_est']}]\n{r['text']}"
                for _, r in grp.iterrows()
            )

            print(f"  [{count}/{total}] {obs_id} ({segment}) ...", end=" ", flush=True)
            val, raw = extract_value(obs, var, context)
            print(f"-> {val!r}")

            results.append({
                "company": COMPANY_NAME, "variable": var_name,
                "obs_id": obs_id, "segment": segment,
                "year": obs.get("year", FISCAL_YEAR),
                "extracted_value": val, "num_chunks_used": len(grp),
                "raw_llm_response": raw,
            })
            time.sleep(2)

    # Saving the results
    df = pd.DataFrame(results)
    df.to_csv(EXTRACTED_PATH, index=False)
    print(f"\n  Saved {len(df)} extractions -> {EXTRACTED_PATH}\n")
    print(df[["obs_id", "variable", "segment", "extracted_value"]].to_string(index=False))


if __name__ == "__main__":
    run()