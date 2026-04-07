"""Tests for the RAG pipeline module."""

from __future__ import annotations

import pytest

from aig_data_rag.pipeline import RAGPipeline


class TestRAGPipeline:
    def test_query_without_ingest_raises(self) -> None:
        pipeline = RAGPipeline(embeddings=object())
        with pytest.raises(RuntimeError, match="No documents ingested"):
            pipeline.query("test")

    def test_initial_chunk_count_is_zero(self) -> None:
        pipeline = RAGPipeline(embeddings=object())
        assert pipeline.chunk_count == 0
