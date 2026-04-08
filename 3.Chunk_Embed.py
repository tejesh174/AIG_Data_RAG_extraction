"""
Chunking & Embedding using PySpark + open ai 

"""

import sys, re, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    RAW_DIR, CHUNKS_DIR, EMBEDDINGS_DIR,
    COMPANY_NAME, YEARS, CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_BACKEND, LOCAL_EMBED_MODEL,
    OPENAI_EMBED_MODEL, OPENAI_API_KEY,
    SPARK_APP_NAME, SPARK_MASTER,
)

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import pandas as pd
import numpy as np


#chunking one file per partition

def chunk_partition(rows):
    import re as _re
    for row in rows:
        company, year, raw_text = row[0], row[1], row[2]
        text = _re.sub(r'<[^>]+>', ' ', raw_text)
        text = _re.sub(r'&nbsp;|&amp;|&lt;|&gt;', ' ', text)
        text = _re.sub(r'\s+', ' ', text).strip()

        size, overlap, cpp = CHUNK_SIZE, CHUNK_OVERLAP, 3000
        chunk_id, start = 0, 0
        while start < len(text):
            end = min(start + size, len(text))
            chunk_text = text[start:end].strip()
            if len(chunk_text) > 50:
                yield (
                    f"{company}_{year}_{chunk_id}",
                    company,
                    int(year),
                    chunk_id,
                    int(start // cpp) + 1,
                    chunk_text,
                )
                chunk_id += 1
            start += size - overlap


# embedding on driver

def embed_on_driver(texts):
    if EMBEDDING_BACKEND == "openai":
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        all_vecs = []
        batch_size = 256 
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for batch_num, i in enumerate(range(0, len(texts), batch_size), 1):
            batch = texts[i:i+batch_size]
            print(f"    Embedding batch {batch_num}/{total_batches} "
                  f"({len(batch)} chunks)...", end=" ", flush=True)
            for attempt in range(1, 6):  #upto 5 retries
                try:
                    resp = client.embeddings.create(
                        input=batch, model=OPENAI_EMBED_MODEL
                    )
                    all_vecs.extend([r.embedding for r in resp.data])
                    print("done")
                    break
                except openai.RateLimitError as e:
                    wait = min(2 ** attempt, 30) 
                    print(f"\n    [RATE LIMIT] Waiting {wait}s (attempt {attempt}/5)...")
                    import time; time.sleep(wait)
            else:
                raise RuntimeError(f"Failed after 5 retries on batch {batch_num}")
            # putting small delay to have some gap between batches on openai
            import time; time.sleep(1.5)
        return np.array(all_vecs, dtype=np.float32)
    else:
        from sentence_transformers import SentenceTransformer
        print(f"  Loading model: {LOCAL_EMBED_MODEL} ...")
        model = SentenceTransformer(LOCAL_EMBED_MODEL)
        print(f"  Embedding {len(texts):,} chunks on driver (CPU ~10-20 mins) ...")
        return model.encode(texts, batch_size=64, show_progress_bar=True)

#main model

def run():
    print(f"\n{'='*60}")
    print(f"  Chunking & Embedding")
    print(f"  PySpark: chunking via mapPartitions")
    print(f"  Driver:  embedding via {EMBEDDING_BACKEND}")
    print(f"  Chunk: {CHUNK_SIZE} | Overlap: {CHUNK_OVERLAP}")
    print(f"{'='*60}\n")

    spark = (
        SparkSession.builder
        .appName(SPARK_APP_NAME)
        .master(SPARK_MASTER)
        .config("spark.driver.memory", "6g")
        .config("spark.rpc.message.maxSize", "512")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    #Loading raw filings on driver
    print("  Loading raw filings ...")
    raw_rows = []
    for year in YEARS:
        path = RAW_DIR / f"{COMPANY_NAME}_{year}.txt"
        if not path.exists():
            print(f"  [WARN] Missing {path.name}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        raw_rows.append((COMPANY_NAME, year, text))
        print(f"  Loaded {path.name} ({len(text):,} chars)")

    if not raw_rows:
        sys.exit("No raw filings found")

    # Chunking via pyspark map partitions
    print(f"\n  Chunking {len(raw_rows)} filings via PySpark mapPartitions ...")
    raw_rdd = spark.sparkContext.parallelize(raw_rows, numSlices=len(raw_rows))
    chunk_rdd = raw_rdd.mapPartitions(chunk_partition)

    chunk_schema = StructType([
        StructField("uid",      StringType(),  False),
        StructField("company",  StringType(),  False),
        StructField("year",     IntegerType(), False),
        StructField("chunk_id", IntegerType(), False),
        StructField("page_est", IntegerType(), False),
        StructField("text",     StringType(),  False),
    ])

    df_chunks = spark.createDataFrame(chunk_rdd, schema=chunk_schema)
    df_chunks.cache()
    chunk_count = df_chunks.count()
    print(f"Total chunks: {chunk_count:,}")

    #Collecting to the driver and saving all chunks
    print(f"\n Collecting chunks to driver....")
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    chunks_path = CHUNKS_DIR / "chunks.csv"
    pdf_chunks = df_chunks.toPandas()
    pdf_chunks.to_csv(chunks_path, index=False)
    print(f" Chunks saved → {chunks_path}")

    #stopping spark before embedding
    spark.stop()
    print(f" Spark stopped.\n")

    #Embedding chunks on driver
    print(f"  Embedding {chunk_count:,} chunks on driver ...")
    vectors = embed_on_driver(pdf_chunks["text"].tolist())

    #embeddings csv
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    embed_path = EMBEDDINGS_DIR / "embeddings.csv"
    pdf_embed = pd.DataFrame({
        "uid": pdf_chunks["uid"].tolist(),
        "embedding_json": [json.dumps([round(float(x), 6) for x in v]) for v in vectors],
    })
    pdf_embed.to_csv(embed_path, index=False)
    print(f"  Embeddings saved to {embed_path}")


if __name__ == "__main__":
    run()