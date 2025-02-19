import librosa
import zarr
from pathlib import Path
from importlib.resources import files
import numpy as np
import time

# import local
import orcAI.auxiliary as aux


def create_spectrogram(wav_file_path,
        spectrogram_parameters = str(files("orcAI.defaults").joinpath("default_spectrogram_parameter.json")),
        verbosity = 1):
    """creates spectrogram according to spectrogram_parameters """
    # Initialize messenger
    msgr = aux.Messenger(verbosity=verbosity)
    msgr.part("Creating spectrogram")
    msgr.info(f"Loading wav file: {wav_file_path}")
    msgr.info(f"NOTE: using channel {spectrogram_parameters['channel']}")
    
    start_time = time.time()
    recording, sr = librosa.load(
        wav_file_path,
        sr=spectrogram_parameters["sampling_rate"],
        mono=False,
    )
    if recording.ndim > 1:
        recording = recording[spectrogram_parameters["channel"] - 1]

    load_time = time.time()
    msgr.info(f"Time for loading wav file: {load_time - start_time:.2f} seconds")

    # create spectrogram
    spectrogram = librosa.stft(
        recording,
        n_fft=spectrogram_parameters["nfft"],
        hop_length=spectrogram_parameters["n_overlap"],
        window="hann",
    )

    frequencies = librosa.fft_frequencies(
        sr=spectrogram_parameters["sampling_rate"], n_fft=spectrogram_parameters["nfft"]
    )

    times = librosa.frames_to_time(
        range(spectrogram.shape[1]),
        sr=spectrogram_parameters["sampling_rate"],
        hop_length=spectrogram_parameters["n_overlap"],
    )
    msgr.info(f"Duration of wav file: {times[-1]:.2f} seconds")

    spectrogram = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)  # Convert to power spectrogram (magnitude squared)
    spec_time = time.time()
    msgr.info(f"Time for generating spectrogram: {spec_time - load_time:.2f} seconds")

    # extract frequency range, clip according to quantiles, and normalise
    freqMinInd = np.argwhere(frequencies <= spectrogram_parameters["freq_range"][0])[0][0]
    freqMaxInd = np.argwhere(frequencies >= spectrogram_parameters["freq_range"][1])[0][0]
    spectrogram = spectrogram[freqMinInd:freqMaxInd, :]

    lower_percentile = np.percentile(
        spectrogram, 100 * spectrogram_parameters["quantiles"][0], method="nearest"
    )
    upper_percentile = np.percentile(
        spectrogram, 100 * spectrogram_parameters["quantiles"][1], method="nearest"
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
    msgr.info(f"  - Time for normalisation: {spec_time - clip_time:.2f} seconds")

    # transpose spectogram
    spectrogram = spectrogram.T

    return spectrogram, frequencies, times


def save_spectrogram(wav_file_path,
        output_dir,
        spectrogram_parameter = str(files("orcAI.defaults").joinpath("default_spectrogram_parameter.json")),
        verbosity = 1):
    """loads wavfile and saves spectrogram to disk according to parameters"""
    msgr = aux.Messenger(verbosity=verbosity)
    spectrogram, frequencies, times = create_spectrogram(wav_file_path, spectrogram_parameter)
    
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
