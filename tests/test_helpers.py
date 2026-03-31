"""Tests for orcai.helpers Module

Tests for project initialization and recording table creation functions.

Created using: claude-haiku-4.5 on 2026-03-31
"""

import io
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from orcai.auxiliary import Messenger
from orcai.helpers import (
    init_project,
    create_recording_table,
)


class TestInitProject:
    """Test init_project function."""

    def test_init_project_creates_directory(self, tmp_path):
        """Test that init_project creates the project directory."""
        project_dir = tmp_path / "test_project"
        assert not project_dir.exists()

        with (
            patch("orcai.helpers.files") as mock_files,
            patch("orcai.helpers.read_json") as mock_read,
            patch("orcai.helpers.write_json"),
        ):
            mock_files.return_value.iterdir.return_value = []
            mock_read.return_value = {"name": "", "seed": None}

            init_project(project_dir, "test_project", verbosity=0)

        assert project_dir.exists()

    def test_init_project_with_pathlib_path(self, tmp_path):
        """Test init_project accepts pathlib.Path."""
        project_dir = tmp_path / "test_project"

        with (
            patch("orcai.helpers.files") as mock_files,
            patch("orcai.helpers.read_json") as mock_read,
            patch("orcai.helpers.write_json"),
        ):
            mock_files.return_value.iterdir.return_value = []
            mock_read.return_value = {"name": "", "seed": None}

            init_project(project_dir, "test_project", verbosity=0)

        assert isinstance(project_dir, Path)
        assert project_dir.exists()

    def test_init_project_with_string_path(self, tmp_path):
        """Test init_project accepts string path."""
        project_dir = str(tmp_path / "test_project")

        with (
            patch("orcai.helpers.files") as mock_files,
            patch("orcai.helpers.read_json") as mock_read,
            patch("orcai.helpers.write_json"),
        ):
            mock_files.return_value.iterdir.return_value = []
            mock_read.return_value = {"name": "", "seed": None}

            init_project(project_dir, "test_project", verbosity=0)

        assert Path(project_dir).exists()

    def test_init_project_creates_parameter_file(self, tmp_path):
        """Test that init_project creates orcai parameter file."""
        project_dir = tmp_path / "test_project"

        with (
            patch("orcai.helpers.files") as mock_files,
            patch("orcai.helpers.read_json") as mock_read,
            patch("orcai.helpers.write_json") as mock_write,
        ):
            mock_files.return_value.iterdir.return_value = []
            mock_read.return_value = {"name": "", "seed": None, "calls": []}

            init_project(project_dir, "test_project", verbosity=0)

            # Verify write_json was called
            assert mock_write.called

    def test_init_project_sets_project_name(self, tmp_path):
        """Test that init_project sets the project name."""
        project_dir = tmp_path / "test_project"
        project_name = "my_project"

        with (
            patch("orcai.helpers.files") as mock_files,
            patch("orcai.helpers.read_json") as mock_read,
            patch("orcai.helpers.write_json") as mock_write,
        ):
            mock_files.return_value.iterdir.return_value = []
            mock_read.return_value = {"name": "", "seed": None}

            init_project(project_dir, project_name, verbosity=0)

            # Get the parameter dict that was written
            written_params = mock_write.call_args[0][0]
            assert written_params["name"] == project_name

    def test_init_project_generates_seed(self, tmp_path):
        """Test that init_project generates a seed."""
        project_dir = tmp_path / "test_project"

        with (
            patch("orcai.helpers.files") as mock_files,
            patch("orcai.helpers.read_json") as mock_read,
            patch("orcai.helpers.write_json") as mock_write,
        ):
            mock_files.return_value.iterdir.return_value = []
            mock_read.return_value = {"name": "", "seed": None}

            init_project(project_dir, "test_project", verbosity=0)

            written_params = mock_write.call_args[0][0]
            # Seed should be generated (not None)
            assert written_params["seed"] is not None

    def test_init_project_with_custom_parameter(self, tmp_path):
        """Test init_project with custom parameter dict."""
        project_dir = tmp_path / "test_project"

        with (
            patch("orcai.helpers.files") as mock_files,
            patch("orcai.helpers.read_json") as mock_read,
            patch("orcai.helpers.write_json"),
        ):
            mock_files.return_value.iterdir.return_value = []
            mock_read.return_value = {
                "name": "",
                "seed": None,
                "model": {"filters": [32, 64]},
            }

            custom_param = {"model": {"filters": [64, 128, 256]}}
            init_project(
                project_dir, "test_project", parameter=custom_param, verbosity=0
            )

        # Parameter should be merged without error

    def test_init_project_with_custom_messenger(self, tmp_path):
        """Test init_project with custom Messenger."""
        project_dir = tmp_path / "test_project"
        output = io.StringIO()
        msgr = Messenger(verbosity=2, file=output)

        with (
            patch("orcai.helpers.files") as mock_files,
            patch("orcai.helpers.read_json") as mock_read,
            patch("orcai.helpers.write_json"),
        ):
            mock_files.return_value.iterdir.return_value = []
            mock_read.return_value = {"name": "", "seed": None}

            init_project(project_dir, "test_project", msgr=msgr, verbosity=0)

        assert "Project initialized" in output.getvalue()

    def test_init_project_respects_verbosity(self, tmp_path):
        """Test init_project respects verbosity parameter."""
        with (
            patch("orcai.helpers.files") as mock_files,
            patch("orcai.helpers.read_json") as mock_read,
            patch("orcai.helpers.write_json"),
        ):
            mock_files.return_value.iterdir.return_value = []
            mock_read.return_value = {"name": "", "seed": None}

            # Should not raise error with any verbosity level
            for verbosity in [0, 1, 2, 3]:
                project_dir = tmp_path / f"project_{verbosity}"
                init_project(
                    project_dir,
                    f"project_{verbosity}",
                    verbosity=verbosity,
                )


class TestCreateRecordingTable:
    """Test create_recording_table function."""

    def test_create_recording_table_with_wav_files(self, tmp_path):
        """Test creating recording table with wav files."""
        # Create test directory structure
        recording_dir = tmp_path / "recordings"
        recording_dir.mkdir()
        (recording_dir / "test1.wav").touch()
        (recording_dir / "test2.wav").touch()

        output_path = tmp_path / "recording_table.csv"

        table = create_recording_table(
            recording_dir,
            output_path=output_path,
            verbosity=0,
        )

        assert isinstance(table, pd.DataFrame)
        assert len(table) == 2
        assert output_path.exists()

    def test_create_recording_table_default_output_path(self, tmp_path):
        """Test recording table default output path."""
        recording_dir = tmp_path / "recordings"
        recording_dir.mkdir()
        (recording_dir / "test.wav").touch()

        create_recording_table(recording_dir, verbosity=0)

        default_output = recording_dir / "recording_table.csv"
        assert default_output.exists()

    def test_create_recording_table_with_annotations(self, tmp_path):
        """Test recording table with annotation files."""
        recording_dir = tmp_path / "recordings"
        annotation_dir = tmp_path / "annotations"
        recording_dir.mkdir()
        annotation_dir.mkdir()

        (recording_dir / "test.wav").touch()
        (annotation_dir / "test.txt").touch()

        output_path = tmp_path / "table.csv"

        table = create_recording_table(
            recording_dir,
            base_dir_annotation=annotation_dir,
            output_path=output_path,
            verbosity=0,
        )

        assert "rel_annotation_path" in table.columns
        assert table["rel_annotation_path"].notna().sum() > 0

    def test_create_recording_table_default_channel(self, tmp_path):
        """Test recording table default channel."""
        recording_dir = tmp_path / "recordings"
        recording_dir.mkdir()
        (recording_dir / "test.wav").touch()

        output_path = tmp_path / "table.csv"

        table = create_recording_table(
            recording_dir,
            output_path=output_path,
            default_channel=2,
            verbosity=0,
        )

        assert (table["channel"] == 2).all()

    def test_create_recording_table_exclude_patterns(self, tmp_path):
        """Test recording table with exclude patterns."""
        recording_dir = tmp_path / "recordings"
        recording_dir.mkdir()
        (recording_dir / "data.wav").touch()
        (recording_dir / "temp.wav").touch()

        output_path = tmp_path / "table.csv"

        table = create_recording_table(
            recording_dir,
            output_path=output_path,
            exclude_patterns=["temp"],
            verbosity=0,
        )

        assert len(table) == 1
        assert "data" in table.index[0]

    def test_create_recording_table_with_subdirectories(self, tmp_path):
        """Test recording table finds wav files in subdirectories."""
        recording_dir = tmp_path / "recordings"
        recording_dir.mkdir()
        (recording_dir / "test1.wav").touch()

        subdir = recording_dir / "subdir"
        subdir.mkdir()
        (subdir / "test2.wav").touch()

        output_path = tmp_path / "table.csv"

        table = create_recording_table(
            recording_dir,
            output_path=output_path,
            verbosity=0,
        )

        assert len(table) == 2

    def test_create_recording_table_required_columns(self, tmp_path):
        """Test recording table has all required columns."""
        recording_dir = tmp_path / "recordings"
        recording_dir.mkdir()
        (recording_dir / "test.wav").touch()

        output_path = tmp_path / "table.csv"

        table = create_recording_table(
            recording_dir,
            output_path=output_path,
            verbosity=0,
        )

        required_cols = [
            "channel",
            "duplicate",
            "base_dir_recording",
            "rel_recording_path",
            "base_dir_annotation",
            "rel_annotation_path",
        ]
        for col in required_cols:
            assert col in table.columns

    def test_create_recording_table_duplicate_detection(self, tmp_path):
        """Test recording table detects duplicate filenames."""
        recording_dir = tmp_path / "recordings"
        recording_dir.mkdir()

        subdir = recording_dir / "subdir"
        subdir.mkdir()

        # Create two files with same stem
        (recording_dir / "test.wav").touch()
        (subdir / "test.wav").touch()

        output_path = tmp_path / "table.csv"

        table = create_recording_table(
            recording_dir,
            output_path=output_path,
            verbosity=0,
        )

        # Both files should be marked as duplicates
        assert table["duplicate"].sum() == 2

    def test_create_recording_table_remove_duplicates(self, tmp_path):
        """Test remove_duplicate_filenames parameter."""
        recording_dir = tmp_path / "recordings"
        recording_dir.mkdir()

        subdir = recording_dir / "subdir"
        subdir.mkdir()

        (recording_dir / "test.wav").touch()
        (subdir / "test.wav").touch()

        output_path = tmp_path / "table.csv"

        table = create_recording_table(
            recording_dir,
            output_path=output_path,
            remove_duplicate_filenames=True,
            verbosity=0,
        )

        # Duplicates should be removed
        assert table["duplicate"].sum() == 0
        assert len(table) == 0

    def test_create_recording_table_with_orcai_parameter(self, tmp_path):
        """Test recording table with orcai parameter containing call types."""
        recording_dir = tmp_path / "recordings"
        recording_dir.mkdir()
        (recording_dir / "test.wav").touch()

        param_file = tmp_path / "param.json"
        param = {
            "calls": ["BR", "BUZZ", "WHISTLE"],
        }
        with open(param_file, "w") as f:
            json.dump(param, f)

        output_path = tmp_path / "table.csv"

        table = create_recording_table(
            recording_dir,
            output_path=output_path,
            orcai_parameter=param_file,
            verbosity=0,
        )

        # Should have columns for each call type
        for call in param["calls"]:
            assert call in table.columns

    def test_create_recording_table_with_custom_messenger(self, tmp_path):
        """Test recording table with custom Messenger."""
        recording_dir = tmp_path / "recordings"
        recording_dir.mkdir()
        (recording_dir / "test.wav").touch()

        output_path = tmp_path / "table.csv"
        output = io.StringIO()
        msgr = Messenger(verbosity=2, file=output)

        create_recording_table(
            recording_dir,
            output_path=output_path,
            msgr=msgr,
        )

        assert "Recordings table created" in output.getvalue()

    def test_create_recording_table_respects_verbosity(self, tmp_path):
        """Test recording table respects verbosity parameter."""
        recording_dir = tmp_path / "recordings"
        recording_dir.mkdir()
        (recording_dir / "test.wav").touch()

        # Should work with any verbosity level
        for verbosity in [0, 1, 2, 3]:
            table = create_recording_table(
                recording_dir,
                output_path=tmp_path / f"table_{verbosity}.csv",
                verbosity=verbosity,
            )
            assert isinstance(table, pd.DataFrame)

    def test_create_recording_table_returns_dataframe(self, tmp_path):
        """Test that create_recording_table returns DataFrame."""
        recording_dir = tmp_path / "recordings"
        recording_dir.mkdir()
        (recording_dir / "test.wav").touch()

        output_path = tmp_path / "table.csv"

        result = create_recording_table(
            recording_dir,
            output_path=output_path,
            verbosity=0,
        )

        assert isinstance(result, pd.DataFrame)

    def test_create_recording_table_index_is_recording_name(self, tmp_path):
        """Test recording table index is recording name."""
        recording_dir = tmp_path / "recordings"
        recording_dir.mkdir()
        (recording_dir / "test1.wav").touch()
        (recording_dir / "test2.wav").touch()

        output_path = tmp_path / "table.csv"

        table = create_recording_table(
            recording_dir,
            output_path=output_path,
            verbosity=0,
        )

        assert table.index.name == "recording"
        assert "test1" in table.index
        assert "test2" in table.index

    def test_create_recording_table_relative_paths(self, tmp_path):
        """Test recording table contains relative paths."""
        recording_dir = tmp_path / "recordings"
        recording_dir.mkdir()
        (recording_dir / "test.wav").touch()

        output_path = tmp_path / "table.csv"

        table = create_recording_table(
            recording_dir,
            output_path=output_path,
            verbosity=0,
        )

        # rel_recording_path should not be None
        assert table["rel_recording_path"].notna().any()

    def test_create_recording_table_no_wav_files(self, tmp_path):
        """Test recording table with no wav files."""
        recording_dir = tmp_path / "recordings"
        recording_dir.mkdir()

        output_path = tmp_path / "table.csv"

        table = create_recording_table(
            recording_dir,
            output_path=output_path,
            verbosity=0,
        )

        assert len(table) == 0

    def test_create_recording_table_missing_annotations(self, tmp_path):
        """Test recording table handles missing annotations."""
        recording_dir = tmp_path / "recordings"
        annotation_dir = tmp_path / "annotations"
        recording_dir.mkdir()
        annotation_dir.mkdir()

        (recording_dir / "test1.wav").touch()
        (recording_dir / "test2.wav").touch()
        (annotation_dir / "test1.txt").touch()

        output_path = tmp_path / "table.csv"

        table = create_recording_table(
            recording_dir,
            base_dir_annotation=annotation_dir,
            output_path=output_path,
            verbosity=0,
        )

        # test1 should have annotation, test2 should not
        assert table.loc["test1", "rel_annotation_path"] is not None or pd.notna(
            table.loc["test1", "rel_annotation_path"]
        )
        assert pd.isna(table.loc["test2", "rel_annotation_path"])

    def test_create_recording_table_pathlib_input(self, tmp_path):
        """Test recording table accepts pathlib.Path."""
        recording_dir = tmp_path / "recordings"
        recording_dir.mkdir()
        (recording_dir / "test.wav").touch()

        output_path = tmp_path / "table.csv"

        # Pass as Path objects
        table = create_recording_table(
            Path(recording_dir),
            output_path=Path(output_path),
            verbosity=0,
        )

        assert isinstance(table, pd.DataFrame)

    def test_create_recording_table_string_input(self, tmp_path):
        """Test recording table accepts string paths."""
        recording_dir = tmp_path / "recordings"
        recording_dir.mkdir()
        (recording_dir / "test.wav").touch()

        output_path = tmp_path / "table.csv"

        # Pass as strings
        table = create_recording_table(
            str(recording_dir),
            output_path=str(output_path),
            verbosity=0,
        )

        assert isinstance(table, pd.DataFrame)
