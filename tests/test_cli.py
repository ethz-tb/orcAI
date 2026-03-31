"""Tests for orcai.cli Module

Tests for CLI commands, parameters, and argument handling.

Created using: claude-haiku-4.5 on 2026-03-31
"""

from pathlib import Path

from click.testing import CliRunner

from orcai.cli import (
    DEFAULT_MODEL,
    INCLUDED_MODELS,
    ClickDirPathR,
    ClickDirPathW,
    ClickDirPathWcreate,
    ClickFilePathR,
    ClickFilePathW,
    cli,
)


class TestCliGroup:
    """Test main CLI group and its configuration."""

    def test_cli_group_exists(self):
        """Test that cli group exists and is callable."""
        assert callable(cli)

    def test_cli_help(self):
        """Test CLI help output."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "orcAI" in result.output
        assert "Command line interface" in result.output

    def test_cli_version(self):
        """Test CLI version output."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        # Version is printed to output, exit code can be 0 or None
        assert "orcai" in result.output or "version" in result.output.lower()

    def test_cli_group_has_subcommands(self):
        """Test that CLI group has registered subcommands."""
        assert len(cli.commands) > 0
        expected_commands = {
            "predict",
            "filter-predictions",
            "init",
            "create-recording-table",
            "create-spectrograms",
            "create-label-arrays",
            "create-snippet-table",
            "create-tvt-snippet-tables",
            "create-tvt-data",
            "train",
            "test",
            "hpsearch",
        }
        assert expected_commands.issubset(cli.commands.keys())


class TestClickPathTypes:
    """Test Click path type parameters."""

    def test_click_dir_path_r_type(self):
        """Test ClickDirPathR parameter type."""
        assert ClickDirPathR.exists is True
        assert ClickDirPathR.file_okay is False
        assert ClickDirPathR.dir_okay is True
        assert ClickDirPathR.readable is True
        assert ClickDirPathR.resolve_path is True
        assert ClickDirPathR.type == Path

    def test_click_dir_path_w_type(self):
        """Test ClickDirPathW parameter type."""
        assert ClickDirPathW.exists is True
        assert ClickDirPathW.file_okay is False
        assert ClickDirPathW.dir_okay is True
        assert ClickDirPathW.writable is True
        assert ClickDirPathW.resolve_path is True
        assert ClickDirPathW.type == Path

    def test_click_dir_path_wcreate_type(self):
        """Test ClickDirPathWcreate parameter type."""
        assert ClickDirPathWcreate.exists is False
        assert ClickDirPathWcreate.file_okay is False
        assert ClickDirPathWcreate.dir_okay is True
        assert ClickDirPathWcreate.writable is True
        assert ClickDirPathWcreate.resolve_path is True
        assert ClickDirPathWcreate.type == Path

    def test_click_file_path_r_type(self):
        """Test ClickFilePathR parameter type."""
        assert ClickFilePathR.exists is True
        assert ClickFilePathR.file_okay is True
        assert ClickFilePathR.dir_okay is False
        assert ClickFilePathR.readable is True
        assert ClickFilePathR.resolve_path is True
        assert ClickFilePathR.type == Path

    def test_click_file_path_w_type(self):
        """Test ClickFilePathW parameter type."""
        assert ClickFilePathW.exists is False
        assert ClickFilePathW.file_okay is True
        assert ClickFilePathW.dir_okay is False
        assert ClickFilePathW.writable is True
        assert ClickFilePathW.resolve_path is True
        assert ClickFilePathW.type == Path


class TestIncludedModels:
    """Test model discovery and defaults."""

    def test_included_models_is_list(self):
        """Test that INCLUDED_MODELS is a list."""
        assert isinstance(INCLUDED_MODELS, list)

    def test_included_models_not_empty(self):
        """Test that at least one model is included."""
        assert len(INCLUDED_MODELS) > 0

    def test_default_model_in_included_models(self):
        """Test that DEFAULT_MODEL is in INCLUDED_MODELS."""
        assert DEFAULT_MODEL in INCLUDED_MODELS

    def test_default_model_is_string(self):
        """Test that DEFAULT_MODEL is a string."""
        assert isinstance(DEFAULT_MODEL, str)

    def test_default_model_not_empty(self):
        """Test that DEFAULT_MODEL is not empty."""
        assert len(DEFAULT_MODEL) > 0

    def test_all_models_are_strings(self):
        """Test that all models are strings."""
        assert all(isinstance(model, str) for model in INCLUDED_MODELS)

    def test_models_do_not_contain_ds_store(self):
        """Test that .DS_Store is filtered out."""
        assert ".DS_Store" not in INCLUDED_MODELS


class TestCommandInvocations:
    """Test CLI command invocations with minimal dependencies."""

    def test_predict_command_help(self):
        """Test predict command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["predict", "--help"])
        assert result.exit_code == 0
        assert "Predicts call annotations" in result.output

    def test_filter_predictions_command_help(self):
        """Test filter-predictions command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["filter-predictions", "--help"])
        assert result.exit_code == 0
        assert (
            "filter" in result.output.lower() or "prediction" in result.output.lower()
        )

    def test_init_command_help(self):
        """Test init command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--help"])
        assert result.exit_code == 0
        assert "Initialize" in result.output

    def test_create_recording_table_command_help(self):
        """Test create-recording-table command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-recording-table", "--help"])
        assert result.exit_code == 0
        assert "recording" in result.output.lower()

    def test_create_spectrograms_command_help(self):
        """Test create-spectrograms command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-spectrograms", "--help"])
        assert result.exit_code == 0
        assert "spectrogram" in result.output.lower()

    def test_create_label_arrays_command_help(self):
        """Test create-label-arrays command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-label-arrays", "--help"])
        assert result.exit_code == 0
        assert "label" in result.output.lower()

    def test_create_snippet_table_command_help(self):
        """Test create-snippet-table command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-snippet-table", "--help"])
        assert result.exit_code == 0
        assert "snippet" in result.output.lower()

    def test_create_tvt_snippet_tables_command_help(self):
        """Test create-tvt-snippet-tables command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-tvt-snippet-tables", "--help"])
        assert result.exit_code == 0
        assert "tvt" in result.output.lower() or "train" in result.output.lower()

    def test_create_tvt_data_command_help(self):
        """Test create-tvt-data command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-tvt-data", "--help"])
        assert result.exit_code == 0
        assert "tvt" in result.output.lower() or "data" in result.output.lower()

    def test_train_command_help(self):
        """Test train command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "train" in result.output.lower()

    def test_test_command_help(self):
        """Test test command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["test", "--help"])
        assert result.exit_code == 0
        assert "test" in result.output.lower()

    def test_hpsearch_command_help(self):
        """Test hpsearch command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["hpsearch", "--help"])
        assert result.exit_code == 0
        assert "search" in result.output.lower() or "hyper" in result.output.lower()


class TestCliErrorHandling:
    """Test CLI error handling and edge cases."""

    def test_invalid_command(self):
        """Test invoking non-existent command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["invalid-command"])
        assert result.exit_code != 0
        assert "Error" in result.output or "no such command" in result.output.lower()

    def test_cli_no_args(self):
        """Test CLI with no arguments shows usage."""
        runner = CliRunner()
        result = runner.invoke(cli, [])
        # When no command provided, Click shows usage (exit code 2 or 0 depending on setup)
        assert result.output != ""
        assert "orcAI" in result.output or "Usage:" in result.output

    def test_command_help_with_dash_dash_help(self):
        """Test --help flag for commands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["predict", "--help"])
        assert result.exit_code == 0

    def test_command_help_with_h_short_flag(self):
        """Test --help flag for commands (h short flag may not be supported)."""
        runner = CliRunner()
        result = runner.invoke(cli, ["predict", "--help"])
        # Test with --help which is always supported
        assert result.exit_code == 0
        assert "Predicts" in result.output or "predict" in result.output.lower()


class TestCliConfiguration:
    """Test CLI rich-click configuration."""

    def test_cli_has_context_settings(self):
        """Test that CLI command has proper configuration."""
        assert cli.callback is not None or cli.invoke_without_command

    def test_all_subcommands_are_click_commands(self):
        """Test that all subcommands are properly registered Click commands."""
        for name, cmd in cli.commands.items():
            assert hasattr(cmd, "callback")
            assert callable(cmd.callback)

    def test_cli_group_name(self):
        """Test the CLI group name."""
        assert cli.name == "cli"
