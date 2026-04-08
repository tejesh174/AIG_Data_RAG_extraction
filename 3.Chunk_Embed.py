"""
Chunking & Embedding using PySpark + OpenAI
"""

import sys, re, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    RAW_DIR, CHUNKS_DIR, EMBEDDINGS_DIR,
    COMPANY_NAME, FISCAL_YEAR, CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_BACKEND, LOCAL_EMBED_MODEL,
    OPENAI_EMBED_MODEL, OPENAI_API_KEY,
    SPARK_APP_NAME, SPARK_MASTER,
)

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import pandas as pd
import numpy as np


# Chunking(runs inside PySpark partitions)
def chunk_partition(rows):
    for row in rows:
        company, year, raw_text = row[0], row[1], row[2]

        # cleaning text
        text = re.sub(r'\s+', ' ', raw_text).strip()

        chunk_id, start = 0, 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk_text = text[start:end].strip()
            if len(chunk_text) > 50:
                yield (
                    f"{company}_{year}_{chunk_id}", 
                    company,
                    int(year),
                    chunk_id,
                    int(start // 3000) + 1,           
                    chunk_text,
                )
                chunk_id += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP


#Embedding (runs on driver)
def embed_on_driver(texts):
    if EMBEDDING_BACKEND == "openai":
        import openai, time
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        all_vecs = []
        for i in range(0, len(texts), 256):
            batch = texts[i:i+256]
            print(f"    Embedding {i}–{i+len(batch)} of {len(texts)} ...", flush=True)
            resp = client.embeddings.create(input=batch, model=OPENAI_EMBED_MODEL)
            all_vecs.extend([r.embedding for r in resp.data])
            time.sleep(1)
        return np.array(all_vecs, dtype=np.float32)
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(LOCAL_EMBED_MODEL)
        return model.encode(texts, batch_size=64, show_progress_bar=True)


# Main Function
def run():
    print(f"\n{'='*60}")
    print(f"  Step 3: Chunking & Embedding")
    print(f"  Chunk: {CHUNK_SIZE} | Overlap: {CHUNK_OVERLAP}")
    print(f"  Embedding: {EMBEDDING_BACKEND}")
    print(f"{'='*60}\n")

    # Loading raw filing
    path = RAW_DIR / f"{COMPANY_NAME}_{FISCAL_YEAR}.txt"
    if not path.exists():
        sys.exit(f"  {path.name} not found. Run Step 1 first.")
    text = path.read_text(encoding="utf-8", errors="replace")
    print(f"  Loaded {path.name} ({len(text):,} chars)")

    # chunking on pyspark
    spark = (SparkSession.builder
             .appName(SPARK_APP_NAME).master(SPARK_MASTER)
             .config("spark.driver.memory", "4g").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    raw_rdd = spark.sparkContext.parallelize([(COMPANY_NAME, FISCAL_YEAR, text)], 1)
    chunk_rdd = raw_rdd.mapPartitions(chunk_partition)

    schema = StructType([
        StructField("uid",      StringType(),  False),
        StructField("company",  StringType(),  False),
        StructField("year",     IntegerType(), False),
        StructField("chunk_id", IntegerType(), False),
        StructField("page_est", IntegerType(), False),
        StructField("text",     StringType(),  False),
    ])

    #chunks data frame
    df_chunks = spark.createDataFrame(chunk_rdd, schema=schema)
    pdf_chunks = df_chunks.toPandas()
    print(f"  Total chunks: {len(pdf_chunks):,}")

    # Saving chunks
    chunks_path = CHUNKS_DIR / "chunks.csv"
    pdf_chunks.to_csv(chunks_path, index=False)
    print(f"  Chunks saved -> {chunks_path}")

    spark.stop()

    # Embedding
    print(f"\n  Embedding {len(pdf_chunks):,} chunks ...")
    vectors = embed_on_driver(pdf_chunks["text"].tolist())

    # Saving embeddings
    embed_path = EMBEDDINGS_DIR / "embeddings.csv"
    pdf_embed = pd.DataFrame({
        "uid": pdf_chunks["uid"].tolist(),
        "embedding_json": [json.dumps([round(float(x), 6) for x in v]) for v in vectors],
    })
    pdf_embed.to_csv(embed_path, index=False)
    print(f"  Embeddings saved to {embed_path}")
    print(f"\n  Done.\n")


if __name__ == "__main__":
    run()