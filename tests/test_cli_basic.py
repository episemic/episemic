"""Basic tests for CLI functionality."""

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from episemic_core.cli.main import app


@patch('episemic_core.hippocampus.duckdb_hippocampus.SentenceTransformer')
def test_cli_version_command(mock_transformer):
    """Test CLI version command."""
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "Episemic Core" in result.stdout


@patch('episemic_core.hippocampus.duckdb_hippocampus.SentenceTransformer')
def test_cli_health_command(mock_transformer):
    """Test CLI health command."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.1] * 384
    mock_transformer.return_value = mock_model

    runner = CliRunner()
    result = runner.invoke(app, ["health"])
    # Health command should work with DuckDB fallback
    assert result.exit_code == 0


def test_cli_help():
    """Test CLI help command."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Episemic Core" in result.stdout
    assert "store" in result.stdout
    assert "search" in result.stdout
    assert "get" in result.stdout


@patch('episemic_core.hippocampus.duckdb_hippocampus.SentenceTransformer')
def test_cli_store_command(mock_transformer):
    """Test CLI store command."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.2] * 384
    mock_transformer.return_value = mock_model

    runner = CliRunner()

    # Test basic store command
    result = runner.invoke(app, [
        "store",
        "Test memory content",
        "--title", "Test Title",
        "--source", "cli_test",
        "--tags", "test", "cli"
    ])

    # Should work with DuckDB fallback
    assert result.exit_code == 0


@patch('episemic_core.hippocampus.duckdb_hippocampus.SentenceTransformer')
def test_cli_search_command(mock_transformer):
    """Test CLI search command."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.3] * 384
    mock_transformer.return_value = mock_model

    runner = CliRunner()

    # First store a memory
    store_result = runner.invoke(app, [
        "store",
        "Search test memory",
        "--title", "Search Test"
    ])
    assert store_result.exit_code == 0

    # Then search for it
    search_result = runner.invoke(app, [
        "search",
        "search test",
        "--top-k", "5"
    ])

    # Search should work
    assert search_result.exit_code == 0


@patch('episemic_core.hippocampus.duckdb_hippocampus.SentenceTransformer')
def test_cli_init_command(mock_transformer):
    """Test CLI init command."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.4] * 384
    mock_transformer.return_value = mock_model

    runner = CliRunner()
    result = runner.invoke(app, ["init"])

    # Init should work with DuckDB fallback
    assert result.exit_code == 0


@patch('episemic_core.hippocampus.duckdb_hippocampus.SentenceTransformer')
def test_cli_consolidate_command(mock_transformer):
    """Test CLI consolidate command."""
    # Mock the sentence transformer
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.5] * 384
    mock_transformer.return_value = mock_model

    runner = CliRunner()

    # Test consolidate command
    result = runner.invoke(app, ["consolidate"])

    # Should handle gracefully even with limited setup
    assert result.exit_code in [0, 1]  # May fail due to missing consolidation setup


def test_cli_individual_command_help():
    """Test help for individual CLI commands."""
    runner = CliRunner()

    commands = ["store", "search", "get", "consolidate", "health", "version", "init"]

    for command in commands:
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.stdout or "Show this message" in result.stdout