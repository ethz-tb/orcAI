from pathlib import Path
from importlib.resources import files
import numpy as np
import time
import pandas as pd
from click import progressbar
from librosa import load, stft, fft_frequencies, frames_to_time, amplitude_to_db

# import local
from orcAI.auxiliary import (
    Messenger,
    read_json,
    save_as_zarr,
    write_vector_to_json,
    resolve_file_paths,
    recording_table_show_func,
)


def make_spectrogram(
    wav_file_path,
    channel=1,
    spectrogram_parameter=str(
        files("orcAI.defaults").joinpath("default_spectrogram_parameter.json")
    ),
    msgr=Messenger(),
):
    """Makes spectrogram from .wav file according to spectrogram_parameter

    Parameters
    ----------
    wav_file_path : (Path | Str)
        Path to the wav file.
    channel : int
        Channel of wav_file to use for the spectrogram.
    spectrogram_parameter : dict | (str | Path)
        Dictionary with parameters for the spectrogram creation or path to JSON containing the same. Defaults to default_spectrogram_parameter.json.
    msgr : Messenger
        Messenger object for logging.

    Returns
    -------
    np.ndarray
        Spectrogram.
    np.ndarray
        Frequencies of the spectrogram.
    np.ndarray
        Times of the spectrogram.
    """

    # TODO: remove channel from spectrogram_parameter and add it as an argument
    msgr.part("Creating spectrogram")

    if isinstance(spectrogram_parameter, (Path | str)):
        spectrogram_parameter = read_json(spectrogram_parameter)

    msgr.info(f"Loading wav file: {wav_file_path.name}")

    start_time = time.time()
    wav_file, _ = load(
        wav_file_path,
        sr=spectrogram_parameter["sampling_rate"],
        mono=False,
    )
    if wav_file.ndim > 1:
        msgr.info(f"Multiple channels found, using channel {channel}")
        wav_file = wav_file[channel - 1]

    load_time = time.time()
    msgr.info(f"Time for loading wav file: {load_time - start_time:.2f} seconds")

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
    spec_time = time.time()
    msgr.info(f"Time for generating spectrogram: {spec_time - load_time:.2f} seconds")

    # extract frequency range, clip according to quantiles, and normalise
    freqMinInd = np.argwhere(frequencies <= spectrogram_parameter["freq_range"][0])[0][
        0
    ]
    freqMaxInd = np.argwhere(frequencies >= spectrogram_parameter["freq_range"][1])[0][
        0
    ]
    spectrogram = spectrogram[freqMinInd:freqMaxInd, :]

    lower_percentile = np.percentile(
        spectrogram, 100 * spectrogram_parameter["quantiles"][0], method="nearest"
    )
    upper_percentile = np.percentile(
        spectrogram, 100 * spectrogram_parameter["quantiles"][1], method="nearest"
    )

    # Clip the spectrogram to the computed percentiles
    start_time = time.time()
    spectrogram = np.clip(spectrogram, lower_percentile, upper_percentile)
    clip_time = time.time()
    msgr.info(f"Time for clipping: {clip_time - start_time:.2f} seconds")

    # Normalize the spectrogram to range [0, 1]
    min_val = np.min(spectrogram)
    max_val = np.max(spectrogram)
    spectrogram = (spectrogram - min_val) / (max_val - min_val)
    spec_time = time.time()
    msgr.info(f"Time for normalisation: {spec_time - clip_time:.2f} seconds")

    # transpose spectogram
    spectrogram = spectrogram.T

    return spectrogram, frequencies, times


def save_spectrogram(
    spectrogram,
    frequencies,
    times,
    output_dir,
    msgr=Messenger(),
):
    """Saves spectrogram as zarr file and frequency and time vectors as json files in output_dir

    Parameters
    ----------
    spectrogram : np.ndarray
        Spectrogram to be saved.
    frequencies : np.ndarray
        Frequencies of the spectrogram.
    times : np.ndarray
        Times of the spectrogram.
    msgr : Messenger
        Messenger object for logging.
    """

    msgr.part("Saving spectrogram")
    # Save spectrogram with Zarr
    save_as_zarr(spectrogram, filename=Path(output_dir, "spectrogram.zarr"), msgr=msgr)

    # save frequency and time vector into json files
    write_vector_to_json(
        frequencies,
        Path(output_dir, "frequencies.json"),
    )
    write_vector_to_json(
        times,
        Path(output_dir, "times.json"),
    )
    return


def create_spectrograms(
    recording_table_path,
    output_dir,
    base_dir=None,
    spectrogram_parameter=files("orcAI.defaults").joinpath(
        "default_spectrogram_parameter.json"
    ),
    label_calls=files("orcAI.defaults").joinpath("default_calls.json"),
    exclude=True,
    verbosity=2,
):
    """Creates spectrograms for all files in spectrogram_table

    Parameters
    ----------
    recording_table_path : Path
        Path to .csv table with columns 'recording', 'channel' and columns indicating possibility of presence of calls (True/False). #TODO: clarify
    base_dir : Path
        Base directory for the wav files. If not None entries in the recording column are interpreted as filenames
        searched for in base_dir and subfolders. If None the entries are interpreted as absolute paths.
    output_dir : Path
        Output directory for the spectrograms. Spectograms are stored in subdirectories named '<recording>/spectrogram'
    spectrogram_parameter : (Path | str) | dict
        Path to the spectrogram parameter file or a dictionary with parameters for the spectrogram creation.
    label_calls : (Path | str) | dict
        Path to a JSON file containing calls for labeling or a dict.
    exclude : bool
        Exclude recordings without possible annotations.
    verbosity : int
        Verbosity level.
    """
    msgr = Messenger(verbosity=verbosity)
    msgr.part("Reading recordings table")
    recording_table = pd.read_csv(recording_table_path)

    if exclude:
        if isinstance(label_calls, (Path | str)):
            label_calls = read_json(label_calls)
        is_included = recording_table[label_calls].apply(lambda x: x.any(), axis=1)
        msgr.info(
            f"Excluded recordings because they lack any possible annotations:", indent=1
        )
        msgr.info(str(recording_table[~is_included]["recording"].values), indent=-1)
        recording_table = recording_table[is_included]

    if base_dir is not None:
        msgr.info(f"Resolving file paths...")
        recording_table["wav_file_path"] = resolve_file_paths(
            base_dir, recording_table["recording"], ".wav", msgr=msgr
        )

    if isinstance(spectrogram_parameter, (Path | str)):
        spectrogram_parameter = read_json(spectrogram_parameter)

    msgr.part(f"Creating {len(recording_table)} spectrograms")
    with progressbar(
        recording_table.index,
        label="Creating spectrograms",
        item_show_func=lambda index: recording_table_show_func(index, recording_table),
    ) as recording_indices:
        for i in recording_indices:
            silent_msgr = Messenger(verbosity=0)
            spectrogram, frequencies, times = make_spectrogram(
                recording_table.loc[i, "wav_file_path"],
                recording_table.loc[i, "channel"],
                spectrogram_parameter,
                msgr=silent_msgr,
            )

            recording_output_dir = Path(output_dir).joinpath(
                recording_table.loc[i, "recording"], "spectrogram"
            )

            save_spectrogram(
                spectrogram,
                frequencies,
                times,
                recording_output_dir,
                msgr=silent_msgr,
            )

    return
