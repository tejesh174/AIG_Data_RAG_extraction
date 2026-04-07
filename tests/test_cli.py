"""Tests for the CLI module."""

from __future__ import annotations

from aig_data_rag.cli import main


class TestCLI:
    def test_info_command(self, capsys: object) -> None:
        main(["info"])
        captured = capsys.readouterr()  # type: ignore[union-attr]
        assert "AIG Data RAG Extraction Pipeline" in captured.out

    def test_no_command_shows_help(self, capsys: object) -> None:
        main([])
        captured = capsys.readouterr()  # type: ignore[union-attr]
        assert "usage" in captured.out.lower() or "AIG" in captured.out
