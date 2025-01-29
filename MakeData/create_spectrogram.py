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
fnstem = Path(wav_file).stem
# test for presence of fnstem in channels_and_labels
if fnstem in channels_and_labels["fnstem"].values:

    # get recording type
    recording_type = list(
        channels_and_labels["recording.type"][channels_and_labels["fnstem"] == fnstem]
    )[0]
    # get audio channel with whale sound recordings
    channel = list(
        channels_and_labels["best.channel"][channels_and_labels["fnstem"] == fnstem]
    )[0]
    spectrogram_dict["channel"] = channel
    # get labels that are present in that annotation file
    # row = channels_and_labels[calls_for_labeling_list][
    #     channels_and_labels["fnstem"] == fnstem
    # ]
    # labels_present = list(row.columns[row.eq("yes").any()])
    print(
        "  - fnstem:",
        fnstem,
        # " labels used:",
        # labels_present,
        " channel:",
        spectrogram_dict["channel"],
    )

    if os.path.exists(wav_file.replace(".wav", ".txt")):
        root_dir_spectrograms = directories_dict[computer]["root_dir_spectrograms"]
    else:
        root_dir_spectrograms = (
            directories_dict[computer]["root_dir_spectrograms"] + "wo_annot/"
        )
    spec.save_spectrogram(wav_file, root_dir_spectrograms, spectrogram_dict)

# %%
print("PROGRAM COMPLETED")
