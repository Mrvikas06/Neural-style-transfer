"""Unit tests for the nst.cli command-line interface entry point."""

from __future__ import annotations

import sys
import pytest

from nst.cli import main
from scripts.run_nst import parse_args


def test_cli_parser_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify that CLI parser displays valid usage when --help is invoked."""
    with pytest.raises(SystemExit) as exc_info:
        parse_args(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Neural Style Transfer" in captured.out
    assert "--content" in captured.out
    assert "--style" in captured.out


def test_cli_parser_required_args() -> None:
    """Verify parser enforces required --content and --style parameters."""
    with pytest.raises(SystemExit):
        parse_args([])
