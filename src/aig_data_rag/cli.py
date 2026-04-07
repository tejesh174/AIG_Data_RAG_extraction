"""CLI entry point for the AIG Data RAG extraction pipeline."""

from __future__ import annotations

import argparse

from aig_data_rag.loader import load_pdf, split_documents


def main(argv: list[str] | None = None) -> None:
    """Run the AIG RAG extraction pipeline from the command line."""
    parser = argparse.ArgumentParser(
        description="AIG 10-K RAG Pipeline - Extract financial data from AIG 10-K filings"
    )
    subparsers = parser.add_subparsers(dest="command")

    # 'info' sub-command: show project info
    subparsers.add_parser("info", help="Show project information")

    # 'ingest' sub-command: load and split a PDF
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a PDF and show chunk statistics")
    ingest_parser.add_argument("filepath", help="Path to a PDF file")
    ingest_parser.add_argument("--chunk-size", type=int, default=1000)
    ingest_parser.add_argument("--chunk-overlap", type=int, default=200)

    args = parser.parse_args(argv)

    if args.command == "info":
        _show_info()
    elif args.command == "ingest":
        _ingest(args.filepath, args.chunk_size, args.chunk_overlap)
    else:
        parser.print_help()


def _show_info() -> None:
    from aig_data_rag import __version__

    print(f"AIG Data RAG Extraction Pipeline v{__version__}")
    print("RAG pipeline for extracting financial data from AIG 10-K filings")
    print()
    print("Available commands:")
    print("  info     - Show this information")
    print("  ingest   - Load a PDF, split into chunks, and show statistics")


def _ingest(filepath: str, chunk_size: int, chunk_overlap: int) -> None:
    print(f"Loading PDF: {filepath}")
    docs = load_pdf(filepath)
    print(f"  Loaded {len(docs)} page(s)")

    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"  Split into {len(chunks)} chunk(s) (size={chunk_size}, overlap={chunk_overlap})")

    if chunks:
        avg_len = sum(len(c.page_content) for c in chunks) / len(chunks)
        print(f"  Average chunk length: {avg_len:.0f} characters")


if __name__ == "__main__":
    main()
