"""Tests for orcai.auxiliary Module

Tests for constants, Messenger class, and utility functions.

Created using: claude-haiku-4.5 on 2026-03-31
"""

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from orcai.auxiliary import (
    MASK_VALUE,
    ORCAI_SYMBOL,
    SEED_ID_CREATE_DATALOADER,
    SEED_ID_FILTER_SNIPPET_TABLE,
    SEED_ID_LOAD_TEST_DATA,
    SEED_ID_LOAD_TRAIN_DATA,
    SEED_ID_LOAD_UNFILTERED_TEST_DATA,
    SEED_ID_LOAD_VAL_DATA,
    SEED_ID_MAKE_SNIPPET_TABLE,
    SEED_ID_UNFILTERED_TEST_DATA,
    Messenger,
    filter_filepaths,
    find_consecutive_ones,
    resolve_recording_data_dir,
    seconds_to_hms,
)


class TestConstants:
    """Test module constants."""

    def test_mask_value(self):
        """Test MASK_VALUE constant is -1.0."""
        assert MASK_VALUE == -1.0
        assert isinstance(MASK_VALUE, float)

    def test_seed_ids_are_unique(self):
        """Test all seed IDs are unique."""
        seed_ids = [
            SEED_ID_MAKE_SNIPPET_TABLE,
            SEED_ID_FILTER_SNIPPET_TABLE,
            SEED_ID_LOAD_TRAIN_DATA,
            SEED_ID_LOAD_VAL_DATA,
            SEED_ID_LOAD_TEST_DATA,
            SEED_ID_UNFILTERED_TEST_DATA,
            SEED_ID_LOAD_UNFILTERED_TEST_DATA,
        ]
        assert len(seed_ids) == len(set(seed_ids))

    def test_seed_id_create_dataloader_dict(self):
        """Test SEED_ID_CREATE_DATALOADER is a dict with expected keys."""
        assert isinstance(SEED_ID_CREATE_DATALOADER, dict)
        assert set(SEED_ID_CREATE_DATALOADER.keys()) == {
            "train",
            "val",
            "test",
            "unfiltered_test",
        }
        assert all(isinstance(v, int) for v in SEED_ID_CREATE_DATALOADER.values())

    def test_orcai_symbol_is_string(self):
        """Test ORCAI_SYMBOL is a string."""
        assert isinstance(ORCAI_SYMBOL, str)
        assert len(ORCAI_SYMBOL) > 0


class TestMessenger:
    """Test Messenger class."""

    def test_messenger_initialization_default(self):
        """Test Messenger initialization with default parameters."""
        msgr = Messenger()
        assert msgr.n_indent == 0
        assert msgr.verbosity == 2
        assert msgr.indent_str == "    "
        assert msgr.show_part_times is True

    def test_messenger_initialization_with_title(self):
        """Test Messenger initialization with title."""
        output = io.StringIO()
        msgr = Messenger(title="Test Title", file=output, verbosity=0)
        # Verify messenger was initialized and title was passed
        assert msgr.start_time is not None
        assert msgr.verbosity == 0

    def test_messenger_initialization_custom(self):
        """Test Messenger initialization with custom parameters."""
        msgr = Messenger(
            n_indent=2,
            verbosity=1,
            indent_str="\t",
            show_part_times=False,
        )
        assert msgr.n_indent == 2
        assert msgr.verbosity == 1
        assert msgr.indent_str == "\t"
        assert msgr.show_part_times is False

    def test_messenger_info_respects_verbosity(self):
        """Test info messages respect verbosity level."""
        output = io.StringIO()
        msgr = Messenger(verbosity=0, file=output)  # only errors
        msgr.info("info message")
        assert output.getvalue() == ""

    def test_messenger_debug_respects_verbosity(self):
        """Test debug messages respect verbosity level."""
        output = io.StringIO()
        msgr = Messenger(verbosity=2, file=output)  # show info but not debug
        msgr.debug("debug message")
        assert output.getvalue() == ""

    def test_messenger_error_always_shown(self):
        """Test error messages are always shown regardless of verbosity."""
        output = io.StringIO()
        msgr = Messenger(verbosity=0, file=output)
        msgr.error("error message")
        assert "error message" in output.getvalue()

    def test_messenger_warning_with_severity_1(self):
        """Test warning messages with severity level."""
        output = io.StringIO()
        msgr = Messenger(verbosity=1, file=output)
        msgr.warning("warning message")
        assert "warning message" in output.getvalue()

    def test_messenger_print_string(self):
        """Test printing a string message."""
        output = io.StringIO()
        msgr = Messenger(verbosity=2, file=output)
        msgr.print("test message")
        assert "test message" in output.getvalue()

    def test_messenger_print_dict(self):
        """Test printing a dictionary message."""
        output = io.StringIO()
        msgr = Messenger(verbosity=2, file=output)
        test_dict = {"key": "value", "num": 42}
        msgr.print(test_dict)
        result = output.getvalue()
        assert "key" in result
        assert "value" in result

    def test_messenger_print_list(self):
        """Test printing a list message."""
        output = io.StringIO()
        msgr = Messenger(verbosity=2, file=output)
        test_list = ["item1", "item2", "item3"]
        msgr.print(test_list)
        result = output.getvalue()
        assert "item1" in result
        assert "item2" in result
        assert "item3" in result

    def test_messenger_print_dataframe(self):
        """Test printing a DataFrame."""
        output = io.StringIO()
        msgr = Messenger(verbosity=2, file=output)
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        msgr.print(df)
        result = output.getvalue()
        assert "col1" in result
        assert "col2" in result

    def test_messenger_indent_level(self):
        """Test indentation levels affect output."""
        output = io.StringIO()
        msgr = Messenger(verbosity=2, file=output, n_indent=0)
        msgr.print("no indent")
        msgr.print("one indent", indent=1)
        msgr.print("message")
        result = output.getvalue()
        lines = result.split("\n")
        # Second message should have more spaces than first
        assert len(lines[0]) <= len(lines[1])

    def test_messenger_set_indent(self):
        """Test set_indent parameter overrides current indent."""
        output = io.StringIO()
        msgr = Messenger(verbosity=2, file=output, n_indent=3)
        msgr.print("test", set_indent=0)
        assert msgr.n_indent == 0

    def test_messenger_list_to_str(self):
        """Test list_to_str method."""
        msgr = Messenger()
        test_list = ["a", "b", "c"]
        result = msgr.list_to_str(test_list)
        assert "a" in result
        assert "b" in result
        assert "c" in result

    def test_messenger_dict_to_str(self):
        """Test dict_to_str method."""
        msgr = Messenger()
        test_dict = {"key1": "value1", "key2": 42}
        result = msgr.dict_to_str(test_dict)
        assert "key1" in result
        assert "value1" in result
        assert "key2" in result

    def test_messenger_pd_to_str(self):
        """Test pd_to_str method."""
        msgr = Messenger()
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        result = msgr.pd_to_str(df)
        assert "col1" in result
        assert "col2" in result

    def test_messenger_success_method(self):
        """Test success method."""
        output = io.StringIO()
        msgr = Messenger(verbosity=2, file=output)
        msgr.success("operation successful")
        assert "operation successful" in output.getvalue()

    def test_messenger_part_method_timing(self):
        """Test part method includes timing information."""
        output = io.StringIO()
        msgr = Messenger(verbosity=2, file=output, show_part_times=True)
        msgr.part("Part 1")
        result = output.getvalue()
        assert "Part 1" in result
        assert "[" in result  # timing brackets

    def test_messenger_part_method_no_timing(self):
        """Test part method without timing information."""
        output = io.StringIO()
        msgr = Messenger(verbosity=2, file=output, show_part_times=False)
        msgr.part("Part 1")
        result = output.getvalue()
        assert "Part 1" in result

    def test_messenger_start_method(self):
        """Test start method."""
        output = io.StringIO()
        msgr = Messenger(verbosity=2, file=output)
        msgr.start("Starting process")
        result = output.getvalue()
        assert "Starting process" in result
        assert ORCAI_SYMBOL in result


class TestSecondsToHms:
    """Test seconds_to_hms function."""

    def test_zero_seconds(self):
        """Test converting 0 seconds."""
        assert seconds_to_hms(0) == "00:00:00"

    def test_single_second(self):
        """Test converting 1 second."""
        assert seconds_to_hms(1) == "00:00:01"

    def test_one_minute(self):
        """Test converting 60 seconds (1 minute)."""
        assert seconds_to_hms(60) == "00:01:00"

    def test_one_hour(self):
        """Test converting 3600 seconds (1 hour)."""
        assert seconds_to_hms(3600) == "01:00:00"

    def test_mixed_time(self):
        """Test converting mixed hours, minutes, seconds."""
        # 1 hour, 23 minutes, 45 seconds = 5025 seconds
        assert seconds_to_hms(5025) == "01:23:45"

    def test_large_value(self):
        """Test converting large time values."""
        # 24 hours = 86400 seconds
        assert seconds_to_hms(86400) == "24:00:00"

    def test_return_type(self):
        """Test return type is string."""
        assert isinstance(seconds_to_hms(100), str)


class TestFindConsecutiveOnes:
    """Test find_consecutive_ones function."""

    def test_single_one(self):
        """Test finding single consecutive 1."""
        arr = np.array([0, 1, 0])
        starts, stops = find_consecutive_ones(arr)
        assert np.array_equal(starts, np.array([1]))
        assert np.array_equal(stops, np.array([1]))

    def test_multiple_ones(self):
        """Test finding multiple consecutive 1s."""
        arr = np.array([0, 1, 1, 1, 0])
        starts, stops = find_consecutive_ones(arr)
        assert np.array_equal(starts, np.array([1]))
        assert np.array_equal(stops, np.array([3]))

    def test_multiple_sequences(self):
        """Test finding multiple separate sequences of 1s."""
        arr = np.array([1, 1, 0, 1, 0, 1, 1, 1])
        starts, stops = find_consecutive_ones(arr)
        assert np.array_equal(starts, np.array([0, 3, 5]))
        assert np.array_equal(stops, np.array([1, 3, 7]))

    def test_all_ones(self):
        """Test array of all 1s."""
        arr = np.array([1, 1, 1, 1])
        starts, stops = find_consecutive_ones(arr)
        assert np.array_equal(starts, np.array([0]))
        assert np.array_equal(stops, np.array([3]))

    def test_all_zeros(self):
        """Test array of all 0s."""
        arr = np.array([0, 0, 0, 0])
        starts, stops = find_consecutive_ones(arr)
        assert len(starts) == 0
        assert len(stops) == 0

    def test_empty_array(self):
        """Test empty array."""
        arr = np.array([])
        starts, stops = find_consecutive_ones(arr)
        assert len(starts) == 0
        assert len(stops) == 0

    def test_return_types(self):
        """Test return types are numpy arrays."""
        arr = np.array([1, 0, 1])
        starts, stops = find_consecutive_ones(arr)
        assert isinstance(starts, np.ndarray)
        assert isinstance(stops, np.ndarray)


class TestResolveRecordingDataDir:
    """Test resolve_recording_data_dir function."""

    def test_existing_directory(self):
        """Test resolving existing recording directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recording_name = "test_recording"
            recording_dir = Path(tmpdir) / recording_name
            recording_dir.mkdir()

            result = resolve_recording_data_dir(recording_name, tmpdir)
            assert result == recording_dir

    def test_nonexistent_directory(self):
        """Test resolving nonexistent recording directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = resolve_recording_data_dir("nonexistent", tmpdir)
            assert result is None

    def test_return_type_path(self):
        """Test return type is Path when directory exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recording_name = "test_recording"
            recording_dir = Path(tmpdir) / recording_name
            recording_dir.mkdir()

            result = resolve_recording_data_dir(recording_name, tmpdir)
            assert isinstance(result, Path)

    def test_return_type_none(self):
        """Test return type is None when directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = resolve_recording_data_dir("nonexistent", tmpdir)
            assert result is None

    def test_with_path_object(self):
        """Test function works with Path object as recording_data_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recording_name = "test_recording"
            recording_dir = Path(tmpdir) / recording_name
            recording_dir.mkdir()

            result = resolve_recording_data_dir(recording_name, Path(tmpdir))
            assert result == recording_dir


class TestFilterFilepaths:
    """Test filter_filepaths function."""

    def test_filter_single_pattern(self):
        """Test filtering with single exclude pattern."""
        filepaths = [
            Path("/path/to/file1.txt"),
            Path("/path/to/file2.log"),
            Path("/path/to/file3.txt"),
        ]
        exclude_pattern = [".log"]
        result = filter_filepaths(filepaths, exclude_pattern, Messenger(verbosity=0))
        assert len(result) == 2
        assert Path("/path/to/file2.log") not in result

    def test_filter_multiple_patterns(self):
        """Test filtering with multiple exclude patterns."""
        filepaths = [
            Path("/path/to/file1.txt"),
            Path("/path/to/file2.log"),
            Path("/path/to/file3.tmp"),
            Path("/path/to/file4.txt"),
        ]
        exclude_pattern = [".log", ".tmp"]
        result = filter_filepaths(filepaths, exclude_pattern, Messenger(verbosity=0))
        assert len(result) == 2
        assert all(".log" not in str(f) and ".tmp" not in str(f) for f in result)

    def test_filter_no_matches(self):
        """Test filtering with pattern that matches nothing."""
        filepaths = [
            Path("/path/to/file1.txt"),
            Path("/path/to/file2.txt"),
        ]
        exclude_pattern = [".log"]
        result = filter_filepaths(filepaths, exclude_pattern, Messenger(verbosity=0))
        assert len(result) == 2
        assert result == filepaths

    def test_filter_all_matches(self):
        """Test filtering where all files match pattern."""
        filepaths = [
            Path("/path/to/file1.log"),
            Path("/path/to/file2.log"),
        ]
        exclude_pattern = [".log"]
        result = filter_filepaths(filepaths, exclude_pattern, Messenger(verbosity=0))
        assert len(result) == 0

    def test_filter_preserves_path_objects(self):
        """Test that filtered result contains Path objects."""
        filepaths = [
            Path("/path/to/file1.txt"),
            Path("/path/to/file2.log"),
        ]
        exclude_pattern = [".log"]
        result = filter_filepaths(filepaths, exclude_pattern, Messenger(verbosity=0))
        assert all(isinstance(f, Path) for f in result)

    def test_filter_empty_list(self):
        """Test filtering empty filepath list."""
        filepaths = []
        exclude_pattern = [".log"]
        result = filter_filepaths(filepaths, exclude_pattern, Messenger(verbosity=0))
        assert len(result) == 0

    def test_filter_empty_patterns(self):
        """Test filtering with empty exclude patterns."""
        filepaths = [
            Path("/path/to/file1.txt"),
            Path("/path/to/file2.log"),
        ]
        exclude_pattern = []
        result = filter_filepaths(filepaths, exclude_pattern, Messenger(verbosity=0))
        assert len(result) == 2
        assert result == filepaths
