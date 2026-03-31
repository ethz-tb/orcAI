"""Tests for json_encoder module.

Tests for JsonEncoderExt custom JSON encoder.

Created using: claude-sonnet-4-6 on 2026-03-31
"""

import json
from pathlib import Path

import numpy as np
import pytest

from orcai.json_encoder import JsonEncoderExt


def encode(obj):
    """Helper to encode and decode a single object via JsonEncoderExt."""
    return json.loads(json.dumps(obj, cls=JsonEncoderExt))


class TestJsonEncoderExtPath:
    """Tests for Path serialization."""

    def test_path_encoded_as_string(self):
        """Path objects are serialized as strings."""
        result = encode(Path("/some/path/file.json"))
        assert result == "/some/path/file.json"
        assert isinstance(result, str)

    def test_relative_path(self):
        """Relative Path objects are serialized as strings."""
        result = encode(Path("relative/path"))
        assert result == "relative/path"

    def test_path_in_dict(self):
        """Path inside a dict is serialized correctly."""
        result = encode({"key": Path("/data/file.txt")})
        assert result == {"key": "/data/file.txt"}

    def test_path_in_list(self):
        """Path inside a list is serialized correctly."""
        result = encode([Path("/a"), Path("/b")])
        assert result == ["/a", "/b"]


class TestJsonEncoderExtNumpy:
    """Tests for numpy float32 serialization."""

    def test_float32_encoded_as_float(self):
        """np.float32 values are serialized as JSON numbers."""
        result = encode(np.float32(3.14))
        assert isinstance(result, float)
        assert abs(result - 3.14) < 1e-5

    def test_float32_zero(self):
        """np.float32 zero is serialized correctly."""
        result = encode(np.float32(0.0))
        assert result == 0.0

    def test_float32_negative(self):
        """Negative np.float32 is serialized correctly."""
        result = encode(np.float32(-1.5))
        assert isinstance(result, float)
        assert abs(result - (-1.5)) < 1e-5

    def test_float32_in_dict(self):
        """np.float32 inside a dict is serialized correctly."""
        result = encode({"loss": np.float32(0.42)})
        assert isinstance(result["loss"], float)

    def test_float32_in_list(self):
        """np.float32 inside a list is serialized correctly."""
        result = encode([np.float32(1.0), np.float32(2.0)])
        assert result == [pytest.approx(1.0), pytest.approx(2.0)]


class TestJsonEncoderExtFallback:
    """Tests for fallback behavior on unsupported types."""

    def test_unsupported_type_raises(self):
        """Unsupported types raise TypeError as per standard JSONEncoder."""
        with pytest.raises(TypeError):
            json.dumps(object(), cls=JsonEncoderExt)

    def test_standard_types_unchanged(self):
        """Standard JSON-serializable types pass through unchanged."""
        data = {"int": 1, "float": 2.5, "str": "hello", "list": [1, 2], "bool": True, "none": None}
        result = encode(data)
        assert result == data
