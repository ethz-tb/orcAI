import librosa
from pathlib import Path
from importlib.resources import files
import numpy as np
import time
import pandas as pd
from click import progressbar

# import local
import orcAI.auxiliary as aux


def create_spectrogram(
    wav_file_path,
    spectrogram_parameter=None,
    spectrogram_parameter_path=str(
        files("orcAI.defaults").joinpath("default_spectrogram_parameter.json")
    ),
    msgr=aux.Messenger(),
):
    """Creates spectrogram from .wav file according to spectrogram_parameter

    Parameters
    ----------
    wav_file_path : Path
        Path to the wav file.
    spectrogram_parameter : dict
        Dictionary with parameters for the spectrogram creation.
    spectrogram_parameter_path : Path
        Path to the spectrogram parameter file. Only used if spectrogram_parameter is None.
    msgr : Messenger
        Messenger object for logging.
    """
    msgr.part("Creating spectrogram")
    # allow for passing either a dict or path to json
    if spectrogram_parameter is None:
        spectrogram_parameter = aux.read_json(spectrogram_parameter_path)
    msgr.info(f"Loading wav file: {wav_file_path}")
    msgr.warning(f"using channel {spectrogram_parameter['channel']}")

    start_time = time.time()
    wav_file, _ = librosa.load(
        wav_file_path,
        sr=spectrogram_parameter["sampling_rate"],
        mono=False,
    )
    if wav_file.ndim > 1:
        wav_file = wav_file[spectrogram_parameter["channel"] - 1]

    load_time = time.time()
    msgr.info(f"Time for loading wav file: {load_time - start_time:.2f} seconds")

    # create spectrogram
    spectrogram = librosa.stft(
        wav_file,
        n_fft=spectrogram_parameter["nfft"],
        hop_length=spectrogram_parameter["n_overlap"],
        window="hann",
    )

    frequencies = librosa.fft_frequencies(
        sr=spectrogram_parameter["sampling_rate"], n_fft=spectrogram_parameter["nfft"]
    )

    times = librosa.frames_to_time(
        range(spectrogram.shape[1]),
        sr=spectrogram_parameter["sampling_rate"],
        hop_length=spectrogram_parameter["n_overlap"],
    )
    msgr.info(f"Duration of wav file: {times[-1]:.2f} seconds")

    spectrogram = librosa.amplitude_to_db(
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
    msgr=aux.Messenger(),
):
    """Loads wavfile and saves spectrogram to disk according to parameters

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
    aux.save_as_zarr(spectrogram, filename=Path(output_dir, "zarr.spc"), msgr=msgr)

    # save frequency and time vector into json files
    aux.write_vector_to_json(
        frequencies,
        Path(output_dir, "frequencies.json"),
    )
    aux.write_vector_to_json(
        times,
        Path(output_dir, "times.json"),
    )
    return


def create_spectrograms(
    recording_table_path,
    base_dir=None,
    output_dir=None,
    spectrogram_parameter=files("orcAI.defaults").joinpath(
        "default_spectrogram_parameter.json"
    ),
    label_calls_path=files("orcAI.defaults").joinpath("default_calls.json"),
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
        Output directory for the spectrograms. If None the spectrograms are saved in the same directory as the wav files.
    spectrogram_parameter : (Path | str) | dict
        Path to the spectrogram parameter file or a dictionary with parameters for the spectrogram creation.
    label_calls_path : Path
        Path to a JSON file containing calls for labeling
    exclude : bool
        Exclude recordings without possible annotations.
    verbosity : int
        Verbosity level.
    """
    msgr = aux.Messenger(verbosity=verbosity)

    recording_table = pd.read_csv(recording_table_path)
    if exclude:
        label_calls = aux.read_json(label_calls_path)
        is_included = recording_table[label_calls].apply(lambda x: x.any(), axis=1)
        msgr.info(
            f"Excluded recordings because they lack any possible annotations:", indent=1
        )
        msgr.info(str(recording_table[~is_included]["recording"].values), indent=-1)
        recording_table = recording_table[is_included]

    if base_dir is not None:
        msgr.info(f"Resolving file paths...")
        recording_table["wav_file_path"] = aux.resolve_file_paths(
            base_dir, recording_table["recording"], ".wav", msgr=msgr
        )

    if isinstance(spectrogram_parameter, (Path | str)):
        spectrogram_parameter = aux.read_json(spectrogram_parameter)

    msgr.part(f"Creating {len(recording_table)} spectrograms")
    with progressbar(
        recording_table.index,
        label="Creating spectrograms",
        item_show_func=lambda index: aux._recording_table_show_func(
            index, recording_table
        ),
    ) as recording_indices:
        for i in recording_indices:
            spectrogram_parameter["channel"] = recording_table.loc[i, "channel"]
            silent_msgr = aux.Messenger(verbosity=0)
            spectrogram, frequencies, times = create_spectrogram(
                recording_table.loc[i, "wav_file_path"],
                spectrogram_parameter,
                msgr=silent_msgr,
            )
            if output_dir is None:
                output_dir = (
                    Path(recording_table.loc[i, "wav_file_path"])
                    .with_suffix("")
                    .joinpath("spectrogram")
                )
            else:
                output_dir = Path(output_dir).joinpath(
                    recording_table.loc[i, "recording"], "spectrogram"
                )

            save_spectrogram(
                spectrogram,
                frequencies,
                times,
                output_dir,
                msgr=silent_msgr,
            )

    return
