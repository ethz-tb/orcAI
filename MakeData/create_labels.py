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
    computer, project_dir = aux.create_labels_commandline_parse()
else:
    computer = "laptop"
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


# %%
# Read all annotation files
print("READING IN ALL ANNOTATION FILES")
annot = pd.DataFrame()
for fn in all_annot_files:
    a = ann.read_annotation_file(fn)
    a = ann.apply_label_dict(a, call_dict)
    annot = pd.concat([annot, a], ignore_index=True)

all_orig_labels = set(annot["origlabel"].unique())
call_dict_keys = set(call_dict.keys())
labels_not_in_call_dict = all_orig_labels.difference(call_dict_keys)
print("labels not in call_dict:", labels_not_in_call_dict)

# %%
# make labels for all files specified below
files = [x.replace(".txt", "") for x in all_annot_files]  # remove extension .txt
for i in range(len(files)):
    fnstem = Path(files[i]).stem
    print("Handling annotation:", files[i], "(", i + 1, "of", len(files), ")")
    # test for presence of fnstem in channels_and_labels
    if fnstem in channels_and_labels["fnstem"].values:
        # get labels that are present in that annotation file
        row = channels_and_labels[calls_for_labeling_list][
            channels_and_labels["fnstem"] == fnstem
        ]
        labels_present = list(row.columns[row.eq("yes").any()])
        print(
            "  - fnstem:",
            fnstem,
            " labels used:",
            labels_present,
        )

        # make label_arr
        mask_value = (
            -1
        )  # set to minus one for label to me masked, set to 0 for label to be assumed to be absent
        root_dir_spectrograms = directories_dict[computer]["root_dir_spectrograms"]
        labels_present = sorted(list(np.unique(labels_present)), key=str)
        # get masked labels
        labels_masked = list(set(calls_for_labeling_list).difference(labels_present))
        if len(labels_present) > 0:  # check if annot file contains any labels used
            try:
                ann.labelarr_from_annot(
                    root_dir_spectrograms,
                    fnstem_annotfile_dict[fnstem],
                    fnstem,
                    call_dict,
                    labels_present,
                    labels_masked,
                    mask_value,
                )
            except:
                print("fnstem not fnstem_annotfile_dict:", fnstem)
        else:
            print("No labels present in:", fn)
            try:
                # Delete the directory and all its contents
                shutil.rmtree(root_dir_spectrograms + fnstem + "/labels/")
                print(f"Directory for '{fnstem}' and all its contents deleted.")
            except FileNotFoundError:
                1 + 1
    else:
        print("WARNING: fnstem not in channels_and_labels:", fnstem)


# %%
annot_files = [x.replace(".txt", "") for x in all_annot_files]  # remove extension .txt
wav_files = [x.replace(".wav", "") for x in all_wav_files]  # remove extension .wav
files_wo_annot = list(set(wav_files) - set(annot_files))


print(f"Printing bash script for create spectrograms: all_specs_{computer}.sh")
with open(project_dir + "all_specs_" + computer + ".sh", "w") as file:
    if computer == "euler":
        file.write(
            "#!/bin/bash\nmodule load stack/2024-06 python/3.11.6 py-pip\nsource myenv/bin/activate\n"
        )
        sbatch_start = 'sbatch --mem-per-cpu=16G --time=20 --wrap="'
        sbatch_end = '"'
    else:
        sbatch_start = ""
        sbatch_end = ""
    for f in annot_files:
        file.write(
            f"{sbatch_start}MakeData/create_spectrogram.py  -c euler -p {project_dir} -w {f}.wav{sbatch_end}\n"
        )


print("PROGRAM COMPLETED")

# %%
