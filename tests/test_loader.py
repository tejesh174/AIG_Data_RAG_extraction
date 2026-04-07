"""Tests for the document loader module."""

from __future__ import annotations

from langchain_core.documents import Document

from aig_data_rag.loader import split_documents


class TestSplitDocuments:
    def test_split_returns_list(self) -> None:
        docs = [Document(page_content="A" * 2000, metadata={"page": 0})]
        result = split_documents(docs, chunk_size=500, chunk_overlap=50)
        assert isinstance(result, list)
        assert len(result) > 1

    def test_split_preserves_metadata(self) -> None:
        docs = [Document(page_content="word " * 400, metadata={"source": "test.pdf", "page": 0})]
        result = split_documents(docs, chunk_size=500, chunk_overlap=50)
        for chunk in result:
            assert "source" in chunk.metadata

    def test_split_small_doc_stays_single(self) -> None:
        docs = [Document(page_content="Short text.", metadata={"page": 0})]
        result = split_documents(docs, chunk_size=1000, chunk_overlap=100)
        assert len(result) == 1
        assert result[0].page_content == "Short text."

    def test_split_empty_list(self) -> None:
        result = split_documents([], chunk_size=500, chunk_overlap=50)
        assert result == []
