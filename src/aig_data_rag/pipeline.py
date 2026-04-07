"""Core RAG pipeline that ties together loading, splitting, embedding, and retrieval."""

from __future__ import annotations

from pathlib import Path

from aig_data_rag.loader import load_pdf, split_documents
from aig_data_rag.vectorstore import build_vectorstore, query_vectorstore


class RAGPipeline:
    """End-to-end RAG pipeline for AIG 10-K financial data extraction."""

    def __init__(self, embeddings: object) -> None:
        self._embeddings = embeddings
        self._vectorstore = None
        self._chunks: list = []

    def ingest(self, filepath: str | Path, chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
        """Ingest a PDF file: load, split, and index into the vector store.

        Returns the number of chunks indexed.
        """
        documents = load_pdf(filepath)
        self._chunks = split_documents(
            documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self._vectorstore = build_vectorstore(self._chunks, self._embeddings)
        return len(self._chunks)

    def query(self, question: str, k: int = 4) -> list:
        """Retrieve the top-k document chunks most relevant to the question."""
        if self._vectorstore is None:
            raise RuntimeError("No documents ingested yet. Call ingest() first.")
        return query_vectorstore(self._vectorstore, question, k=k)

    @property
    def chunk_count(self) -> int:
        """Return the number of chunks currently indexed."""
        return len(self._chunks)
