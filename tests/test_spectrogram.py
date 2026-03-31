"""Tests for spectrogram module.

Tests for calculate_spectrogram, preprocess_spectrogram, and save_spectrogram.

Created using: claude-sonnet-4-6 on 2026-03-31
"""

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from orcai.spectrogram import calculate_spectrogram, preprocess_spectrogram, save_spectrogram


# ---------------------------------------------------------------------------
# calculate_spectrogram
# ---------------------------------------------------------------------------


class TestCalculateSpectrogram:
    """Tests for calculate_spectrogram with a synthetic numpy audio array."""

    def _make_audio(self, n_samples: int = 8000) -> np.ndarray:
        """Return a 1-D float32 audio array (1 second at 8 kHz)."""
        rng = np.random.default_rng(0)
        return rng.standard_normal(n_samples).astype(np.float32)

    def test_output_shapes(self, spectrogram_parameter):
        """Returns (spectrogram, frequencies, times) with consistent shapes."""
        audio = self._make_audio()
        spec, freqs, times = calculate_spectrogram(audio, channel=1, spectrogram_parameter=spectrogram_parameter)
        n_fft_bins = spectrogram_parameter["nfft"] // 2 + 1
        assert spec.shape[0] == n_fft_bins
        assert freqs.shape[0] == n_fft_bins
        assert spec.shape[1] == times.shape[0]

    def test_spectrogram_dtype_float(self, spectrogram_parameter):
        """Spectrogram values are floating point."""
        audio = self._make_audio()
        spec, _, _ = calculate_spectrogram(audio, channel=1, spectrogram_parameter=spectrogram_parameter)
        assert np.issubdtype(spec.dtype, np.floating)

    def test_multichannel_selects_correct_channel(self, spectrogram_parameter):
        """Passing a 2-D (channels x samples) array uses the requested channel."""
        rng = np.random.default_rng(1)
        audio_2ch = rng.standard_normal((2, 8000)).astype(np.float32)
        spec_ch1, _, _ = calculate_spectrogram(
            audio_2ch, channel=1, spectrogram_parameter=spectrogram_parameter
        )
        spec_ch2, _, _ = calculate_spectrogram(
            audio_2ch, channel=2, spectrogram_parameter=spectrogram_parameter
        )
        # Different channels should yield different spectrograms
        assert not np.allclose(spec_ch1, spec_ch2)

    def test_frequencies_cover_nyquist(self, spectrogram_parameter):
        """Frequency axis spans 0 to Nyquist (sr/2)."""
        audio = self._make_audio()
        _, freqs, _ = calculate_spectrogram(audio, channel=1, spectrogram_parameter=spectrogram_parameter)
        assert freqs[0] == pytest.approx(0.0)
        assert freqs[-1] == pytest.approx(spectrogram_parameter["sampling_rate"] / 2)


# ---------------------------------------------------------------------------
# preprocess_spectrogram
# ---------------------------------------------------------------------------


class TestPreprocessSpectrogram:
    """Tests for preprocess_spectrogram."""

    def test_output_transposed(self, synthetic_spectrogram_raw, spectrogram_frequencies, spectrogram_parameter):
        """Output spectrogram is transposed (time x freq) relative to input (freq x time)."""
        n_freq_in, n_time = synthetic_spectrogram_raw.shape
        spec_out, _ = preprocess_spectrogram(
            synthetic_spectrogram_raw, spectrogram_frequencies, spectrogram_parameter
        )
        n_freq_expected = (
            np.argwhere(spectrogram_frequencies >= spectrogram_parameter["freq_range"][1])[0][0]
            - np.argwhere(spectrogram_frequencies <= spectrogram_parameter["freq_range"][0])[0][0]
        )
        assert spec_out.shape == (n_time, n_freq_expected)

    def test_output_normalized_0_1(self, synthetic_spectrogram_raw, spectrogram_frequencies, spectrogram_parameter):
        """Output values are clipped and normalized to [0, 1]."""
        spec_out, _ = preprocess_spectrogram(
            synthetic_spectrogram_raw, spectrogram_frequencies, spectrogram_parameter
        )
        assert spec_out.min() >= 0.0
        assert spec_out.max() <= 1.0

    def test_frequency_range_applied(self, synthetic_spectrogram_raw, spectrogram_frequencies, spectrogram_parameter):
        """Frequency range is applied: output has fewer bins and max is bounded by freq_range[1]."""
        _, freqs_out = preprocess_spectrogram(
            synthetic_spectrogram_raw, spectrogram_frequencies, spectrogram_parameter
        )
        # freq_max_i is the first index >= freq_range[1], so all output freqs are < freq_range[1]
        assert freqs_out.max() < spectrogram_parameter["freq_range"][1]
        # Output should be a strict subset of input frequencies
        assert len(freqs_out) < len(spectrogram_frequencies)

    def test_quantile_clipping_reduces_range(self, spectrogram_frequencies, spectrogram_parameter):
        """Quantile clipping reduces the dynamic range of extreme values."""
        # Create spectrogram with one very large outlier
        rng = np.random.default_rng(42)
        spec = rng.standard_normal((spectrogram_frequencies.shape[0], 50)).astype(np.float32)
        spec[0, 0] = 1e6  # extreme outlier

        spec_out, _ = preprocess_spectrogram(spec, spectrogram_frequencies, spectrogram_parameter)
        # After clipping + normalization the max should still be 1.0 (not inflated)
        assert spec_out.max() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# save_spectrogram
# ---------------------------------------------------------------------------


class TestSaveSpectrogram:
    """Tests for save_spectrogram."""

    def test_creates_zarr_file(self, tmp_path, synthetic_spectrogram_raw, spectrogram_frequencies):
        """zarr spectrogram file is created in output_dir."""
        times = np.linspace(0, 1, synthetic_spectrogram_raw.shape[1])
        save_spectrogram(synthetic_spectrogram_raw, spectrogram_frequencies, times, tmp_path)
        assert (tmp_path / "spectrogram.zarr").exists()

    def test_creates_frequencies_json(self, tmp_path, synthetic_spectrogram_raw, spectrogram_frequencies):
        """frequencies.json is written to output_dir."""
        times = np.linspace(0, 1, synthetic_spectrogram_raw.shape[1])
        save_spectrogram(synthetic_spectrogram_raw, spectrogram_frequencies, times, tmp_path)
        freq_file = tmp_path / "frequencies.json"
        assert freq_file.exists()

    def test_creates_times_json(self, tmp_path, synthetic_spectrogram_raw, spectrogram_frequencies):
        """times.json is written to output_dir."""
        times = np.linspace(0, 1, synthetic_spectrogram_raw.shape[1])
        save_spectrogram(synthetic_spectrogram_raw, spectrogram_frequencies, times, tmp_path)
        assert (tmp_path / "times.json").exists()

    def test_zarr_shape_matches_input(self, tmp_path, synthetic_spectrogram_raw, spectrogram_frequencies):
        """Saved zarr array has the same shape as the input spectrogram."""
        times = np.linspace(0, 1, synthetic_spectrogram_raw.shape[1])
        save_spectrogram(synthetic_spectrogram_raw, spectrogram_frequencies, times, tmp_path)
        arr = zarr.open_array(tmp_path / "spectrogram.zarr", mode="r")
        assert arr.shape == synthetic_spectrogram_raw.shape

    def test_times_json_content(self, tmp_path, synthetic_spectrogram_raw, spectrogram_frequencies):
        """times.json has min, max, length keys matching the times array."""
        n = synthetic_spectrogram_raw.shape[1]
        times = np.linspace(0.0, 2.0, n)
        save_spectrogram(synthetic_spectrogram_raw, spectrogram_frequencies, times, tmp_path)
        data = json.loads((tmp_path / "times.json").read_text())
        assert data["min"] == pytest.approx(0.0, abs=1e-5)
        assert data["max"] == pytest.approx(2.0, abs=1e-5)
        assert data["length"] == n
