"""
Retrieve top-k relevant chunks per variable and year.

"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    EMBEDDINGS_DIR, CHUNKS_DIR, DATA_DIR, RESULTS_DIR,
    VARIABLES, TOP_K_PAGES, CHUNK_SIZE,
    EMBEDDING_BACKEND, LOCAL_EMBED_MODEL, OPENAI_EMBED_MODEL, OPENAI_API_KEY,
    COMPANY_NAME, YEARS,
)

RETRIEVED_PATH = RESULTS_DIR / "retrieved_chunks.csv"


# Lexical terms - aligned with actual AIG 10-K section headers
LEXICAL_TERMS = {
    "total_revenues": [
        "total revenues",
        "total revenue",
        "premiums",
        "net investment income",
        "consolidated statements of income",
        "consolidated results of operations",
        "selected consolidated financial data",
    ],
    "net_income": [
        "net income (loss) attributable to aig",
        "net income attributable to aig",
        "net loss attributable to aig",
        "net income (loss)",
        "consolidated statements of income",
        "consolidated results of operations",
        "consolidated statements of comprehensive income",
    ],
    "industry": [
        "general insurance",
        "life and retirement",
        "operating structure",
        "core businesses",
        "business segments",
        "diversified mix of businesses",
        "reportable segments",
    ],
}

def embed_texts(texts):
    """Embedding a list of texts using the configured backend """
    if EMBEDDING_BACKEND == "openai":
        if not OPENAI_API_KEY:
            sys.exit("OPENAI_API_KEY is required when EMBEDDING_BACKEND='openai'.")
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        all_vecs = []
        for i in range(0, len(texts), 512):
            resp = client.embeddings.create(
                input=texts[i:i+512], model=OPENAI_EMBED_MODEL
            )
            all_vecs.extend([item.embedding for item in resp.data])
        return np.array(all_vecs, dtype=np.float32)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(LOCAL_EMBED_MODEL)
    return model.encode(texts, batch_size=64, show_progress_bar=False)

#cosine similarity between query vector and matrix of embeddings
def cosine_sim(q_vec, matrix):
    q = q_vec / (np.linalg.norm(q_vec) + 1e-9)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    return (matrix / norms) @ q

#Small deterministic boost for exact SEC table labels and segment names.
def lexical_boost(texts, var_name):
    terms = LEXICAL_TERMS.get(var_name, [])
    boosts = []
    for text in texts:
        text_l = str(text).lower()
        hits = sum(1 for term in terms if term in text_l)
        boosts.append(min(0.15, hits * 0.03))
    return np.array(boosts, dtype=np.float32)


def run():
    print(f"\n{'='*60}")
    print(f"  Retrieval  (TOP_K={TOP_K_PAGES} per year/query)")
    print(f"  Company: {COMPANY_NAME}  |  Years: {YEARS}")
    print(f"  Embedding backend: {EMBEDDING_BACKEND}")
    print(f"{'='*60}\n")

    # Loading chunks produced before
    chunks_path = CHUNKS_DIR / "chunks.csv"
    if not chunks_path.exists():
        sys.exit(f"Chunks file not found")

    print("  Loading chunks ...")
    df_chunks = pd.read_csv(chunks_path)
    print(f" Loaded {len(df_chunks):,} chunks")

    # Loading embeddings
    embed_path = EMBEDDINGS_DIR / "embeddings.csv"
    if not embed_path.exists():
        sys.exit(f"Embeddings file not found")

    print("  Loading embeddings ...")
    df_embed = pd.read_csv(embed_path)
    df_embed["embedding"] = df_embed["embedding_json"].apply(json.loads)
    print(f"  Loaded {len(df_embed):,} embeddings")

    # Merging chunks with embeddings
    df = df_chunks.merge(df_embed[["uid", "embedding"]], on="uid").reset_index(drop=True)
    matrix = np.array(df["embedding"].tolist(), dtype=np.float32)
    print(f" Merged matrix: {matrix.shape[0]:,} chunks x {matrix.shape[1]} dims\n")

    # Retreiving for each variable
    all_retrieved = []

    for var in VARIABLES:
        var_name = var["name"]
        var_display = var.get("display", var_name)
        print(f"  Variable: {var_display} ({var_name})")

        query_vecs = embed_texts(var["search_queries"])

        for year, df_year in df.groupby("year", sort=True):
            year_idx = df_year.index.to_numpy()
            year_matrix = matrix[year_idx]
            year_boost = lexical_boost(df_year["text"].tolist(), var_name)

            for query_text, q_vec in zip(var["search_queries"], query_vecs):
                semantic_scores = cosine_sim(q_vec, year_matrix)
                scores = semantic_scores + year_boost
                top_local_idx = np.argsort(scores)[::-1][:TOP_K_PAGES]

                for rank, local_idx in enumerate(top_local_idx):
                    idx = year_idx[local_idx]
                    row = df.iloc[idx]
                    all_retrieved.append({
                        "company":        row["company"],
                        "year":           int(row["year"]),
                        "variable":       var_name,
                        "query":          query_text,
                        "chunk_uid":      row["uid"],
                        "page_est":       int(row["page_est"]),
                        "score":          float(scores[local_idx]),
                        "semantic_score": float(semantic_scores[local_idx]),
                        "lexical_boost":  float(year_boost[local_idx]),
                        "rank":           rank + 1,
                        "text":           row["text"],
                    })

                top_pages = [int(df.iloc[year_idx[i]]["page_est"])
                             for i in top_local_idx]
                print(f"    {year} | '{query_text[:55]}' -> pages {top_pages}")
        print()

    #Deduplicating and saving
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_ret = (
        pd.DataFrame(all_retrieved)
        .sort_values("score", ascending=False)
        .drop_duplicates(subset=["company", "year", "variable", "chunk_uid"])
        .reset_index(drop=True)
    )
    df_ret.to_csv(RETRIEVED_PATH, index=False)
    print(f"  Saved {len(df_ret):,} retrieved chunks -> {RETRIEVED_PATH}\n")

    # Retrieval Summary
    print("  Retrieval summary:")
    for var in VARIABLES:
        vn = var["name"]
        subset = df_ret[df_ret["variable"] == vn]
        avg_score = subset["score"].mean() if len(subset) > 0 else 0
        print(f"    {vn}: {len(subset)} chunks, avg score {avg_score:.4f}")
    print()


if __name__ == "__main__":
    run()
