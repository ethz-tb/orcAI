"""Integration tests for predict.

Run with real data:
    uv run pytest -m integration --wav-file /path/to/file.wav

Created using: claude-sonnet-4-6 on 2026-03-31
"""

import re

import pytest

from orcai.predict import predict


@pytest.mark.integration
def test_predict_wav(wav_file, channel, tmp_path, test_messenger):
    """predict() runs end-to-end on a real wav file and produces a non-empty output file."""
    output_file = tmp_path / "predictions.txt"
    msgr, output = test_messenger

    predict(
        recording_path=wav_file,
        channel=channel,
        output_path=output_file,
        msgr=msgr,
    )

    assert output_file.exists(), "No output file was created"
    assert output_file.stat().st_size > 0, "Output file is empty"
    assert re.search(r"found \d+ acoustic signals", output.getvalue()), (
        "Expected 'found <n> acoustic signals' in output"
    )
    assert re.search(r"Prediction finished", output.getvalue()), (
        "Expected 'Prediction finished' in output"
    )
