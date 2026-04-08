"""
Retrieve top-k relevant chunks per observation.

"""

import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    EMBEDDINGS_DIR, CHUNKS_DIR, RESULTS_DIR,
    VARIABLES, OBSERVATIONS, TOP_K_CHUNKS,
    EMBEDDING_BACKEND, LOCAL_EMBED_MODEL,
    OPENAI_EMBED_MODEL, OPENAI_API_KEY,
    COMPANY_NAME, FISCAL_YEAR,
)

RETRIEVED_PATH = RESULTS_DIR / "retrieved_chunks.csv"

# Lexical boost terms per variable
LEXICAL_TERMS = {
    "net_premiums_written": [
        "net premiums written", "underwriting results",
        "business segment operations",
        "Consolidated Results of Operations", "international commercial", "global personal",
    ],
    "underwriting_income": [
    "Underwriting income", "combined ratio",
    "business segment operations","years ended december 31",
    "international commercial", "global personal", "underwriting income loss",
],
    "regulatory_jurisdiction": [
        "regulatory regimes", "financial services agency",
        "insurance regulator", "prudential supervisor", "monetary authority",
        "insurance operations", "lead state regulator",
    ],
}

# Lexical boost for chunks containing key terms
def lexical_boost(texts, var_name, obs=None):
    terms = LEXICAL_TERMS.get(var_name, [])
    if obs:
        segment = obs.get("segment", obs.get("jurisdiction", ""))
        if segment:
            terms = terms + [segment.lower()]
        year = obs.get("year", "")
        if year:
            terms = terms + [str(year)]
    boosts = []
    for text in texts:
        text_l = str(text).lower()
        hits = sum(1 for term in terms if term in text_l)
        boosts.append(min(0.20, hits * 0.03))
    return np.array(boosts, dtype=np.float32)

# Embedding query texts
def embed_texts(texts):
    if EMBEDDING_BACKEND == "openai":
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        all_vecs = []
        for i in range(0, len(texts), 512):
            resp = client.embeddings.create(
                input=texts[i:i+512], model=OPENAI_EMBED_MODEL
            )
            all_vecs.extend([item.embedding for item in resp.data])
        return np.array(all_vecs, dtype=np.float32)
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(LOCAL_EMBED_MODEL)
        return model.encode(texts, batch_size=64, show_progress_bar=False)


# Cosine similarity
def cosine_sim(q_vec, matrix):
    q = q_vec / (np.linalg.norm(q_vec) + 1e-9)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    return (matrix / norms) @ q


# Main Function
def run():
    print(f"\n{'='*60}")
    print(f"  Step 4: Retrieval (TOP_K={TOP_K_CHUNKS})")
    print(f"  Company: {COMPANY_NAME} | Filing: {FISCAL_YEAR} 10-K")
    print(f"  Embedding: {EMBEDDING_BACKEND}")
    print(f"{'='*60}\n")

    # Loading chunks
    chunks_path = CHUNKS_DIR / "chunks.csv"
    if not chunks_path.exists():
        sys.exit(f"Chunks not found")
    df_chunks = pd.read_csv(chunks_path)
    print(f"  Loaded {len(df_chunks):,} chunks")

    # Loading embeddings
    embed_path = EMBEDDINGS_DIR / "embeddings.csv"
    if not embed_path.exists():
        sys.exit(f"Embeddings not found")
    df_embed = pd.read_csv(embed_path)
    df_embed["embedding"] = df_embed["embedding_json"].apply(json.loads)
    print(f"  Loaded {len(df_embed):,} embeddings")

    # Merging chunks with embeddings
    df = df_chunks.merge(df_embed[["uid", "embedding"]], on="uid").reset_index(drop=True)
    matrix = np.array(df["embedding"].tolist(), dtype=np.float32)
    print(f"  Matrix: {matrix.shape[0]:,} chunks x {matrix.shape[1]} dims\n")

    # Retrieving for each variable and observation
    all_retrieved = []

    for var in VARIABLES:
        var_name = var["name"]
        print(f"  Variable: {var['display']} ({var_name})")
        obs_list = OBSERVATIONS.get(var_name, [])
        for obs in obs_list:
            obs_id = obs["obs_id"]
            segment = obs.get("segment", obs.get("jurisdiction", ""))
            year = obs.get("year", "")
            obs_queries = []
            for base_query in var["search_queries"]:
                obs_queries.append(f"{base_query} {segment} {year}".strip())
            obs_query_vecs = embed_texts(obs_queries)
            boost = lexical_boost(df["text"].tolist(), var_name, obs)
            best_scores = np.zeros(len(df), dtype=np.float32)
            best_query = [""] * len(df)
            for query_text, q_vec in zip(obs_queries, obs_query_vecs):
                semantic = cosine_sim(q_vec, matrix)
                combined = semantic + boost
                for idx in range(len(df)):
                    if combined[idx] > best_scores[idx]:
                        best_scores[idx] = combined[idx]
                        best_query[idx] = query_text
            top_idx = np.argsort(best_scores)[::-1][:TOP_K_CHUNKS]
            for rank, idx in enumerate(top_idx):
                row = df.iloc[idx]
                all_retrieved.append({
                    "company":    row["company"],
                    "variable":   var_name,
                    "obs_id":     obs_id,
                    "chunk_uid":  row["uid"],
                    "page_est":   int(row["page_est"]),
                    "score":      float(best_scores[idx]),
                    "best_query": best_query[idx],
                    "rank":       rank + 1,
                    "text":       row["text"],
                })
            top_pages = [int(df.iloc[i]["page_est"]) for i in top_idx]
            print(f"    {obs_id} ({segment}) -> pages {top_pages}")
        print()

    # Saving retrieved chunks
    df_ret = (
        pd.DataFrame(all_retrieved)
        .sort_values("score", ascending=False)
        .drop_duplicates(subset=["variable", "obs_id", "chunk_uid"])
        .reset_index(drop=True)
    )
    df_ret.to_csv(RETRIEVED_PATH, index=False)
    print(f"  Saved {len(df_ret):,} retrieved chunks -> {RETRIEVED_PATH}")

    # Retrieval Summary
    print("\n  Retrieval summary:")
    for var in VARIABLES:
        vn = var["name"]
        subset = df_ret[df_ret["variable"] == vn]
        avg_score = subset["score"].mean() if len(subset) > 0 else 0
        print(f"    {vn}: {len(subset)} chunks, avg score {avg_score:.4f}")
    print()


if __name__ == "__main__":
    run()