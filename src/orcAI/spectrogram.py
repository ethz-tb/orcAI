from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
from librosa import amplitude_to_db, fft_frequencies, frames_to_time, load, stft
from tqdm import tqdm

from orcAI.auxiliary import (
    Messenger,
)
from orcAI.io import read_json, save_as_zarr, write_vector_to_json


def make_spectrogram(
    wav_file_path: Path | str,
    channel: int = 1,
    orcai_parameter: Path | str = files("orcAI.defaults").joinpath(
        "default_orcai_parameter.json"
    ),
    verbosity: int = 2,
    msgr: Messenger | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Makes spectrogram from .wav file according to orcai_parameter

    Parameters
    ----------
    wav_file_path : (Path | Str)
        Path to the wav file.
    channel : int
        Channel of wav_file to use for the spectrogram.
    orcai_parameter : dict | (Path | str)
        Dictionary with orcai parameters or path to JSON containing the same. Defaults to default_orcai_parameter.json.
    verbosity : int
        Verbosity level. 0: only errors, 1: only warnings, 2: info, 3: debug.
    msgr : Messenger
        Messenger object for logging. If None, a new Messenger object is created.

    Returns
    -------
    np.ndarray
        Spectrogram.
    np.ndarray
        Frequencies of the spectrogram.
    np.ndarray
        Times of the spectrogram.
    """
    if msgr is None:
        msgr = Messenger(verbosity=verbosity, title="Making spectrogram")

    if isinstance(orcai_parameter, (Path | str)):
        orcai_parameter = read_json(orcai_parameter)
    spectrogram_parameter = orcai_parameter["spectrogram"]

    msgr.part(
        f"Loading & resampling (to {spectrogram_parameter['sampling_rate'] / 1000:.2f} kHz) wav file: {wav_file_path.stem}"
    )

    wav_file, _ = load(
        wav_file_path,
        sr=spectrogram_parameter["sampling_rate"],
        mono=False,
    )
    if wav_file.ndim > 1:
        msgr.info(f"Multiple channels found, using channel {channel}")
        wav_file = wav_file[channel - 1]

    msgr.part("Calculating power spectrogram by STFT")

    # create spectrogram
    spectrogram = stft(
        wav_file,
        n_fft=spectrogram_parameter["nfft"],
        hop_length=spectrogram_parameter["n_overlap"],
        window="hann",
    )

    frequencies = fft_frequencies(
        sr=spectrogram_parameter["sampling_rate"], n_fft=spectrogram_parameter["nfft"]
    )

    times = frames_to_time(
        range(spectrogram.shape[1]),
        sr=spectrogram_parameter["sampling_rate"],
        hop_length=spectrogram_parameter["n_overlap"],
    )
    msgr.info(f"Duration of wav file: {times[-1]:.2f} seconds")

    spectrogram = amplitude_to_db(
        np.abs(spectrogram), ref=np.max
    )  # Convert to power spectrogram (magnitude squared)

    msgr.part("Extracting frequency range and clipping spectrogram")

    # extract frequency range, clip according to quantiles, and normalise
    freq_min_i = np.argwhere(frequencies <= spectrogram_parameter["freq_range"][0])[0][
        0
    ]
    freq_max_i = np.argwhere(frequencies >= spectrogram_parameter["freq_range"][1])[0][
        0
    ]
    spectrogram = spectrogram[freq_min_i:freq_max_i, :]

    lower_percentile = np.percentile(
        spectrogram, 100 * spectrogram_parameter["quantiles"][0], method="nearest"
    )
    upper_percentile = np.percentile(
        spectrogram, 100 * spectrogram_parameter["quantiles"][1], method="nearest"
    )

    # Clip the spectrogram to the computed percentiles
    spectrogram = np.clip(spectrogram, lower_percentile, upper_percentile)

    msgr.part("Normalizing spectrogram")

    # Normalize the spectrogram to range [0, 1]
    min_val = np.min(spectrogram)
    max_val = np.max(spectrogram)
    spectrogram = (spectrogram - min_val) / (max_val - min_val)

    # transpose spectogram
    spectrogram = spectrogram.T
    msgr.success("Spectrogram created.")
    return spectrogram, frequencies, times


def save_spectrogram(
    spectrogram: np.ndarray,
    frequencies: np.ndarray,
    times: np.ndarray,
    output_dir: Path | str,
    verbosity: int = 2,
    msgr: Messenger | None = None,
) -> None:
    """Saves spectrogram as zarr file and frequency and time vectors as json files in output_dir

    Parameters
    ----------
    spectrogram : np.ndarray
        Spectrogram to be saved.
    frequencies : np.ndarray
        Frequencies of the spectrogram.
    times : np.ndarray
        Times of the spectrogram.
    output_dir : Path | str
        Output directory for the spectrogram.
    verbosity : int
        Verbosity level. 0: only errors, 1: only warnings, 2: info, 3: debug.
    msgr : Messenger
        Messenger object for logging. If None, a new Messenger object is created.

    Returns
    -------
    None
        Saves spectrogram in output_dir
    """
    if msgr is None:
        msgr = Messenger(verbosity=verbosity, title="Saving spectrogram")

    msgr.part("Saving spectrogram")
    # Save spectrogram with Zarr
    save_as_zarr(spectrogram, filename=Path(output_dir, "spectrogram.zarr"))

    # save frequency and time vector into json files
    write_vector_to_json(
        frequencies,
        Path(output_dir, "frequencies.json"),
    )
    write_vector_to_json(
        times,
        Path(output_dir, "times.json"),
    )
    msgr.success("Spectrogram saved.")
    return


def _make_and_save_spectrogram(recording_info, orcai_parameter, output_dir):
    """Helper function for creating and saving spectrograms for a single recording"""
    silent_msgr = Messenger(verbosity=0)

    spectrogram, frequencies, times = make_spectrogram(
        Path(recording_info.base_dir_recording).joinpath(
            recording_info.rel_recording_path
        ),
        recording_info.channel,
        orcai_parameter,
        msgr=silent_msgr,
    )

    recording_output_dir = Path(output_dir).joinpath(
        recording_info.recording, "spectrogram"
    )

    save_spectrogram(
        spectrogram,
        frequencies,
        times,
        recording_output_dir,
        msgr=silent_msgr,
    )
    return recording_info.recording


def create_spectrograms(
    recording_table_path: Path | str,
    output_dir: Path | str,
    base_dir_recording: Path | str | None = None,
    orcai_parameter: Path | str | None = files("orcAI.defaults").joinpath(
        "default_orcai_parameter.json"
    ),
    include_not_annotated: bool = False,
    include_no_possible_annotations: bool = False,
    overwrite: bool = False,
    verbosity: int = 2,
    msgr: Messenger | None = None,
) -> None:
    """Creates spectrograms for all files in recording table at recording_table_path

    Parameters
    ----------
    recording_table_path : Path | str
        Path to .csv table with columns 'recording', 'channel', 'base_dir_recording',
        'rel_recording_path' and columns indicating possibility of presence of calls (True/False).
    output_dir : Path | str
        Output directory for the spectrograms. Spectograms are stored in subdirectories named '<recording>/spectrogram'
    base_dir_recording : Path
        Base directory for the wav files. If None the base_dir_recording is taken from the recording_table.
    orcai_parameter : (Path | str) | dict
        Path to the orcai parameter file or a dictionary with parameters.
    include_not_annotated: bool
        Include recordings without annotations.
    include_no_possible_annotations : bool
        Include recordings without possible annotations.
    overwrite : bool
        Recreate existing spectrograms.
    verbosity : int
        Verbosity level. 0: only errors, 1: only warnings, 2: info, 3: debug.
    msgr : Messenger
        Messenger object for logging. If None, a new Messenger object is created.


    Returns
    -------
    None
        Saves spectrograms in output_dir
    """
    if msgr is None:
        msgr = Messenger(verbosity=verbosity, title="Creating spectrograms")

    msgr.part("Reading recordings table")
    recording_table = pd.read_csv(recording_table_path)
    output_dir = Path(output_dir)

    if isinstance(orcai_parameter, (Path | str)):
        orcai_parameter = read_json(orcai_parameter)

    if not include_not_annotated:
        not_annotated = recording_table["base_dir_annotation"].isna()
        if len(not_annotated) > 0:
            msgr.info(
                f"Excluded {not_annotated.sum()} recordings because they are not annotated."
            )
            recording_table = recording_table[~not_annotated]

    if not include_no_possible_annotations:
        label_calls = orcai_parameter["calls"]
        is_included = recording_table[label_calls].apply(lambda x: x.any(), axis=1)
        if sum(~is_included) > 0:
            msgr.info(
                "Excluded recordings because they lack any possible annotations:",
                indent=1,
            )
            msgr.info(str(recording_table[~is_included]["recording"].values), indent=-1)
            recording_table = recording_table[is_included]

    if not overwrite:
        existing_spectrograms = recording_table["recording"].apply(
            lambda x: output_dir.joinpath(x, "spectrogram").exists()
        )
        if sum(existing_spectrograms) > 0:
            msgr.info(
                f"Skipping {sum(existing_spectrograms)} recordings because they already have spectrograms."
            )
            recording_table = recording_table[~existing_spectrograms]

    if base_dir_recording is not None:
        recording_table["base_dir_recording"] = base_dir_recording

    msgr.part(f"Creating {len(recording_table)} spectrograms")

    for recording in tqdm(
        recording_table.itertuples(index=False),
        desc="Making spectrograms",
        total=len(recording_table),
    ):
        _make_and_save_spectrogram(recording, orcai_parameter, output_dir)

    msgr.success("Spectrograms created.")
    return
