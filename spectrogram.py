# %%
# import
import librosa
import zarr
from pathlib import Path
import numpy as np
import numpy as np
import pandas as pd
import time

# import local
import auxiliary as aux
import spectrogram as spec


# %%
# functions
def create_spectrogram(wav_file, spectrogram_dict):
    """creates spectrogram according to parameters in spectrogram_dict"""
    print("  - creating spectrogram")
    start_time = time.time()
    recording, sr = librosa.load(
        wav_file,
        sr=spectrogram_dict["sampling_rate"],
        mono=False,
    )
    if recording.ndim > 1:
        recording = recording[spectrogram_dict["channel"] - 1]

    load_time = time.time()
    print(f"     - Time for loading wav file: {load_time - start_time:.2f} seconds")

    # use matplotlib to generate spectrogram (OLD and slow version)
    # Sxx, f, t, im = plt.specgram(
    #     recording,
    #     NFFT=parameters_dict['nfft'],
    #     Fs=rate,
    #     noverlap=int(parameters_dict['nfft'] / 2),
    #     # cmap="gray",
    # )
    # freqMinInd = np.argwhere(f < parameters_dict['freq_range'][0])[0][0]
    # freqMaxInd = np.argwhere(f > parameters_dict['freq_range'][1])[0][0]
    # spectrogram = np.log10(Sxx[freqMinInd:freqMaxInd, :] + 10**-200)

    # create spectrogram
    spectrogram = librosa.stft(
        recording,
        n_fft=spectrogram_dict["nfft"],
        hop_length=spectrogram_dict["n_overlap"],
        window="hann",
    )

    frequencies = librosa.fft_frequencies(
        sr=spectrogram_dict["sampling_rate"], n_fft=spectrogram_dict["nfft"]
    )

    times = librosa.frames_to_time(
        range(spectrogram.shape[1]),
        sr=spectrogram_dict["sampling_rate"],
        hop_length=spectrogram_dict["n_overlap"],
    )

    spectrogram = librosa.amplitude_to_db(
        np.abs(spectrogram), ref=np.max
    )  # Convert to power spectrogram (magnitude squared)
    spec_time = time.time()
    print(
        f"     - Time for generating spectrogram: {spec_time - load_time:.2f} seconds"
    )

    # extract frequency range, clip according to quantiles, and normalise
    freqMinInd = np.argwhere(frequencies <= spectrogram_dict["freq_range"][0])[0][0]
    freqMaxInd = np.argwhere(frequencies >= spectrogram_dict["freq_range"][1])[0][0]
    spectrogram = spectrogram[freqMinInd:freqMaxInd, :]

    lower_percentile = np.percentile(
        spectrogram, 100 * spectrogram_dict["quantiles"][0], method="nearest"
    )
    upper_percentile = np.percentile(
        spectrogram, 100 * spectrogram_dict["quantiles"][1], method="nearest"
    )

    # Clip the spectrogram to the computed percentiles
    start_time = time.time()
    spectrogram = np.clip(spectrogram, lower_percentile, upper_percentile)
    clip_time = time.time()
    print(f"     - Time for clipping: {clip_time - start_time:.2f} seconds")

    # Normalize the spectrogram to range [0, 1]

    min_val = np.min(spectrogram)
    max_val = np.max(spectrogram)
    spectrogram = (spectrogram - min_val) / (max_val - min_val)
    spec_time = time.time()
    print(f"     - Time for normalisation: {spec_time - clip_time:.2f} seconds")

    # transpose spectogram
    spectrogram = spectrogram.T

    return spectrogram, frequencies, times


def save_spectrogram(wav_file, root_dir, spectrogram_dict):
    """loads wavfile and saves spectrogram to disk according to parameters"""
    spectrogram, frequencies, times = create_spectrogram(wav_file, spectrogram_dict)

    # Save spectrogram with Zarr
    start_time = time.time()
    stem_dir = Path(wav_file).stem + "/"
    sub_dir = "spectrogram/"
    zarr_fn = "/zarr.spc"
    zarr_file = zarr.open(
        root_dir + stem_dir + sub_dir + zarr_fn,
        mode="w",
        shape=spectrogram.shape,
        chunks=(2000, spectrogram.shape[1]),
        dtype="float32",
        compressor=zarr.Blosc(cname="zlib"),
    )
    zarr_file[:] = spectrogram
    save_time = time.time()
    print(f"     - Time for for saving to disk: {save_time - start_time:.2f} seconds")

    # save frequency and time vector into json files
    aux.write_vector_to_json(
        frequencies,
        root_dir + stem_dir + sub_dir + "frequencies.json",
    )
    aux.write_vector_to_json(
        times,
        root_dir + stem_dir + sub_dir + "times.json",
    )
    return
