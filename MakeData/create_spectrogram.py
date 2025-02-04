#!/usr/bin/env python

# %%

# import
import numpy as np
import pandas as pd
from pathlib import Path
import os
import shutil
import sys


# import local
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import spectrogram as spec
import auxiliary as aux
import annotation as ann


# %%
# Read command line if interactive
interactive = aux.check_interactive()
if not interactive:
    print("Command-line call:", " ".join(sys.argv))
    (
        computer,
        project_dir,
        wav_file,
    ) = aux.create_spectrogram_commandline_parse()
else:
    computer = "laptop"
    wav_file = "/Volumes/OrcAI-Disk/Acoustics/2023_dtag/oo23_181a093.wav"
    project_dir = "/Users/sb/polybox/Documents/Research/Sebastian/OrcAI_project/"


# %%
# Read parameters
print("Project directory:", project_dir)
os.chdir(project_dir)

print("READ IN PARAMETERS")
dicts = {
    "directories_dict": "GenericParameters/directories.dict",
    "call_dict": "GenericParameters/call.dict",
    "spectrogram_dict": "GenericParameters/spectrogram.dict",
    "calls_for_labeling_list": "GenericParameters/calls_for_labeling.list",
}
for key, value in dicts.items():
    print("  - reading", key)
    globals()[key] = aux.read_dict(value, True)

print("  - reading in channels_and_labels.csv")
channels_and_labels = pd.read_csv(
    project_dir + "GenericParameters/channels_and_labels.csv"
)


# %%
# Make a single or all spectrograms
if wav_file == "all_annotated_files":
    # %%
    # read in .txt and .wav files
    print(
        "READING .txt AND .wav FILE NAMES AND ELIMINATING FILES THAT SHOULD NOT BE INCLUDED"
    )

    eliminate = [
        "._",
        "_ChB",
        "_Chb",
        "Movie",
        "Norway",
        "_acceleration",
        "_depthtemp",
        "_H.",
        "_orig",
        "_old",
    ]

    all_annot_files, fnstem_annotfile_dict, all_wav_files, fnstem_wavfile_dict = (
        aux.wav_and_annot_files(computer, directories_dict, eliminate)
    )
    wav_file_list = [x.replace(".txt", ".wav") for x in all_annot_files]

else:
    wav_file_list = [wav_file]

for i in range(len(wav_file_list)):
    fnstem = Path(wav_file_list[i]).stem
    print("fnstem", i, "of", len(wav_file_list), ":", fnstem)
    # test for presence of fnstem in channels_and_labels
    if fnstem in channels_and_labels["fnstem"].values:

        # get recording type
        recording_type = list(
            channels_and_labels["recording.type"][
                channels_and_labels["fnstem"] == fnstem
            ]
        )[0]
        # get audio channel with whale sound recordings
        channel = list(
            channels_and_labels["best.channel"][channels_and_labels["fnstem"] == fnstem]
        )[0]
        spectrogram_dict["channel"] = channel

        print(
            "  - fnstem:",
            fnstem,
            " channel:",
            spectrogram_dict["channel"],
        )

        root_dir_spectrograms = directories_dict[computer]["root_dir_spectrograms"]
        spec.save_spectrogram(wav_file_list[i], root_dir_spectrograms, spectrogram_dict)

# %%
print("PROGRAM COMPLETED")
