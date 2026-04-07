"""FAISS-based vector store for document retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_community.vectorstores import FAISS

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings


def build_vectorstore(documents: list[Document], embeddings: Embeddings) -> FAISS:
    """Build a FAISS vector store from a list of documents and an embeddings model."""
    return FAISS.from_documents(documents, embeddings)


def query_vectorstore(
    vectorstore: FAISS,
    query: str,
    k: int = 4,
) -> list[Document]:
    """Query the vector store and return the top-k most relevant documents."""
    return vectorstore.similarity_search(query, k=k)
