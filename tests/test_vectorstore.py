"""Tests for the vector store module."""

from __future__ import annotations

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from aig_data_rag.vectorstore import build_vectorstore, query_vectorstore


class FakeEmbeddings(Embeddings):
    """Deterministic fake embeddings for testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        return rng.randn(64).tolist()


class TestVectorStore:
    def test_build_and_query(self) -> None:
        docs = [
            Document(page_content="AIG reported revenue of $50 billion.", metadata={"page": 1}),
            Document(page_content="The insurance segment grew by 12%.", metadata={"page": 2}),
            Document(page_content="Net income was $4.3 billion.", metadata={"page": 3}),
        ]
        embeddings = FakeEmbeddings()
        vs = build_vectorstore(docs, embeddings)
        results = query_vectorstore(vs, "What was AIG revenue?", k=2)
        assert len(results) == 2
        assert all(isinstance(d, Document) for d in results)

    def test_query_returns_k_results(self) -> None:
        docs = [
            Document(page_content=f"Document {i} content.", metadata={"page": i}) for i in range(10)
        ]
        embeddings = FakeEmbeddings()
        vs = build_vectorstore(docs, embeddings)
        results = query_vectorstore(vs, "test query", k=3)
        assert len(results) == 3
