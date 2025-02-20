import librosa
import zarr
from pathlib import Path
from importlib.resources import files
import numpy as np
import time
import pandas as pd
import click

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
    recording, sr = librosa.load(
        wav_file_path,
        sr=spectrogram_parameter["sampling_rate"],
        mono=False,
    )
    if recording.ndim > 1:
        recording = recording[spectrogram_parameter["channel"] - 1]

    load_time = time.time()
    msgr.info(f"Time for loading wav file: {load_time - start_time:.2f} seconds")

    # create spectrogram
    spectrogram = librosa.stft(
        recording,
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
    wav_file_path,
    output_dir=None,
    spectrogram_parameter=None,
    spectrogram_parameter_path=str(
        files("orcAI.defaults").joinpath("default_spectrogram_parameter.json")
    ),
    msgr=aux.Messenger(),
):
    """Loads wavfile and saves spectrogram to disk according to parameters

    Parameters
    ----------
    wav_file_path : Path
        Path to the wav file.
    output_dir : Path
        Output directory for the spectrograms. If None the spectrograms are saved in the same directory as the wav files.
    spectrogram_parameter : dict
        Dictionary with parameters for the spectrogram creation.
    spectrogram_parameter_path : Path
        Path to the spectrogram parameter file. Only used if spectrogram_parameter is None.
    msgr : Messenger
        Messenger object for logging.
    """

    if output_dir is None:
        output_dir = Path(wav_file_path).parent

    # allow for passing either a dict or path to json
    if spectrogram_parameter is None:
        spectrogram_parameter = aux.read_json(spectrogram_parameter_path)

    # Create spectrogram
    spectrogram, frequencies, times = create_spectrogram(
        wav_file_path, spectrogram_parameter=spectrogram_parameter, msgr=msgr
    )

    msgr.part("OrcAI - saving spectrogram")
    # Save spectrogram with Zarr
    start_time = time.time()
    stem = Path(wav_file_path).stem
    sub_dir = "spectrogram"
    zarr_file_name = "zarr.spc"
    zarr_file = zarr.open(
        str(Path(output_dir, stem, sub_dir, zarr_file_name)),
        mode="w",
        shape=spectrogram.shape,
        chunks=(2000, spectrogram.shape[1]),
        dtype="float32",
        compressor=zarr.Blosc(cname="zlib"),
    )
    zarr_file[:] = spectrogram
    save_time = time.time()
    msgr.info(f"Time for for saving to disk: {save_time - start_time:.2f} seconds")

    # save frequency and time vector into json files
    aux.write_vector_to_json(
        frequencies,
        Path(output_dir, stem, sub_dir, "frequencies.json"),
    )
    aux.write_vector_to_json(
        times,
        Path(output_dir, stem, sub_dir, "times.json"),
    )
    return


@click.command(
    help="Creates spectrograms for all files in spectrogram_table",
    short_help="Creates spectrograms for all files in spectrogram_table",
    no_args_is_help=True,
    epilog="For further information visit: https://gitlab.ethz.ch/seb/orcai_test",
)
@click.option(
    "--wav_table_path",
    "-st",
    type=aux.ClickFilePathR,
    required=True,
    help="Path to .csv table with columns 'wav_file', 'channel' and columns corresponding to calls intendend for teaching indicating possibility of presence of calls.",
)
@click.option(
    "--base_dir",
    "-bd",
    type=aux.ClickDirPathR,
    default=None,
    show_default="None",
    help="Base directory for the wav files. If not None entries in the wav_file column are interpreted as filenames searched for in base_dir and subfolders. If None the entries are interpreted as absolute paths.",
)
@click.option(
    "--output_dir",
    "-od",
    type=aux.ClickDirPathW,
    default=None,
    show_default="None",
    help="Output directory for the spectrograms. If None the spectrograms are saved in the same directory as the wav files.",
)
@click.option(
    "--spectrogram_parameter_path",
    "-sp",
    type=aux.ClickFilePathR,
    default=files("orcAI.defaults").joinpath("default_spectrogram_parameter.json"),
    show_default="default_spectrogram_parameter.json",
    help="Path to the spectrogram parameter file.",
)
@click.option(
    "--verbosity",
    "-v",
    type=click.IntRange(0, 2),
    default=1,
    show_default=True,
    help="Verbosity level.",
)
def create_spectrograms(
    wav_table_path,
    base_dir=None,
    output_dir=None,
    spectrogram_parameter_path=files("orcAI.defaults").joinpath(
        "default_spectrogram_parameter.json"
    ),
    verbosity=2,
):
    """Creates spectrograms for all files in spectrogram_table

    Parameters
    ----------
    wav_table_path : Path
        Path to .csv table with columns 'wav_file', 'channel' and columns corresponding to calls intendend for
        teaching indicating possibility of presence of calls.
    base_dir : Path
        Base directory for the wav files. If not None entries in the wav_file column are interpreted as filenames
        searched for in base_dir and subfolders. If None the entries are interpreted as absolute paths.
    output_dir : Path
        Output directory for the spectrograms. If None the spectrograms are saved in the same directory as the wav files.
    spectrogram_parameter_path : Path
        Path to the spectrogram parameter file.
    msgr : Messenger
        Messenger object for logging.
    """
    msgr = aux.Messenger(verbosity=verbosity)
    msgr.part("OrcAI - Creating spectrograms")
    spectrogram_table = pd.read_csv(wav_table_path)
    if base_dir is not None:
        msgr.info(f"Resolving file paths...")
        spectrogram_table["wav_file"] = [
            list(Path(base_dir).rglob(wav_file))[0]
            for wav_file in spectrogram_table["wav_file"]
        ]

    spectrogram_parameter = aux.read_json(spectrogram_parameter_path)

    with click.progressbar(
        range(len(spectrogram_table["wav_file"])),
        label="Creating spectrograms",
        item_show_func=lambda i: _progress_show_func(i, spectrogram_table),
    ) as spectrograms:
        for i_wav_file in spectrograms:
            spectrogram_parameter["channel"] = spectrogram_table["best.channel"][
                i_wav_file
            ]
            save_spectrogram(
                spectrogram_table["wav_file"][i_wav_file],
                output_dir,
                spectrogram_parameter,
                msgr=aux.Messenger(verbosity=0),
            )

    return


def _progress_show_func(i, spectrogram_table):
    if i is not None:
        return (
            spectrogram_table["wav_file"][i].name
            + ", Ch "
            + str(spectrogram_table["best.channel"][i])
        )
