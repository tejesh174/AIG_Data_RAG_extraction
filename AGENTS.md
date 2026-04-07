# AGENTS.md

## Cursor Cloud specific instructions

This is a Python project using `pyproject.toml` with setuptools. The source code lives in `src/aig_data_rag/`.

### Quick reference

| Action | Command |
|--------|---------|
| Install (editable + dev) | `pip install -e ".[dev]"` |
| Lint | `python3 -m ruff check src/ tests/` |
| Format check | `python3 -m ruff format --check src/ tests/` |
| Auto-format | `python3 -m ruff format src/ tests/` |
| Tests | `python3 -m pytest tests/ -v` |
| Run CLI | `python3 -m aig_data_rag.cli info` |

### Notes

- The system Python is `python3` (not `python`). Use `python3` for all commands.
- The package installs to `~/.local` (user site-packages) since the system Python is managed by the OS.
- Tests use `FakeEmbeddings` from `tests/test_vectorstore.py` to avoid needing an LLM API key for vector store tests.
- The RAG pipeline requires a real embeddings provider (e.g., OpenAI) for production use; tests use deterministic fake embeddings based on numpy RNG.
