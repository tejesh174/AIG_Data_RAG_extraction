"""Document loader for AIG 10-K PDF filings."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

if TYPE_CHECKING:
    from langchain_core.documents import Document


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def load_pdf(filepath: str | Path) -> list[Document]:
    """Load a PDF file and return a list of Document objects (one per page)."""
    loader = PyPDFLoader(str(filepath))
    return loader.load()


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into smaller chunks suitable for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(documents)
