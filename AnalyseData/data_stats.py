#!/usr/bin/env python

# import

import pandas as pd
import numpy as np
import librosa
from pathlib import Path


# import local
import auxiliary as aux
import annotation as ann


# %%
# Read command line if interactive
interactive = aux.check_interactive()
sbatch = True
if not interactive:
    computer = aux.data_stats_commandline_parse()
else:
    computer = "laptop"

# %%
print("READ IN PARAMETERS")
dicts = {
    "directories_dict": "directories.dict",
    "calls_for_labeling_list": "calls_for_labeling.list",
    "call_dict": "call.dict",
}
for key, value in dicts.items():
    print("  - reading", key)
    globals()[key] = aux.read_dict(value, True)


# %%
# data stats
intervals = False
if intervals:
    train_df = pd.read_csv(
        directories_dict["laptop"]["root_dir_tvtdata"] + "train.csv.gz"
    )
    val_df = pd.read_csv(directories_dict["laptop"]["root_dir_tvtdata"] + "val.csv.gz")
    test_df = pd.read_csv(
        directories_dict["laptop"]["root_dir_tvtdata"] + "test.csv.gz"
    )
    df = pd.concat([train_df, val_df, test_df])

    def compute_intervals(df, fns):
        # Step 1: Sort by row_start
        tdf = (
            df[df["fnstem_path"] == fns]
            .sort_values(by="row_start")
            .reset_index(drop=True)
        )

        # Step 2: Merge overlapping intervals
        merged_intervals = []
        current_start, current_stop = tdf.iloc[0]["row_start"], tdf.iloc[0]["row_stop"]

        for _, row in tdf.iterrows():
            if row["row_start"] <= current_stop:  # Overlap or adjacent
                current_stop = max(current_stop, row["row_stop"])  # Merge intervals
            else:  # No overlap
                merged_intervals.append([current_start, current_stop])
                current_start, current_stop = row["row_start"], row["row_stop"]

        # Append the last interval
        merged_intervals.append([current_start, current_stop])

        # Step 3: Identify gaps between merged intervals
        no_signal_intervals = []
        for i in range(len(merged_intervals) - 1):
            gap_start = merged_intervals[i][1] + 1
            gap_stop = merged_intervals[i + 1][0] - 1
            if gap_start <= gap_stop:
                no_signal_intervals.append((gap_start, gap_stop))

        return np.asarray(merged_intervals), np.asarray(no_signal_intervals)

    print("Data stats:")
    for i in range(len(df)):
        max_stretch_of_all = 0
        merged_intervals, no_signal_intervals = compute_intervals(
            df, df["fnstem_path"].iloc[i]
        )
        print(" - fnstem:", df["fnstem_path"].iloc[i])
        print(" - # of fragments:", len(merged_intervals))
        lengths_overlapping_signal = merged_intervals[:, 1] - merged_intervals[:, 0]
        max_stretch = np.max(lengths_overlapping_signal)
        print("  - longest stretch:", max_stretch / 376 * 4)
        if max_stretch_of_all < max_stretch:
            max_stretch_of_all = max_stretch
        lengths_no_signal = no_signal_intervals[:, 1] - no_signal_intervals[:, 0]
        print(
            "  - % covered",
            sum(lengths_overlapping_signal)
            / (np.max(merged_intervals[:, 1]) - np.min(merged_intervals[:, 0])),
        )


# %%
# %%
# read in .txt and .wav files
audio_stats = True
if audio_stats:
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

    durations = []
    dirnames = []
    fnstems = []
    for f in all_annot_files:
        durations += [librosa.get_duration(path=f.replace(".txt", ".wav"))]
        dirnames += [Path(f).parts[-2]]
        fnstems += [Path(f).stem]

    total_recording = pd.DataFrame(
        {"dirname": dirnames, "fnstem": fnstems, "duration": durations}
    )

    tmp = total_recording[["dirname", "duration"]].groupby(["dirname"]).sum()
    hms = [aux.seconds_to_hms(x) for x in tmp["duration"]]
    summary_total_recording = pd.DataFrame(
        {
            "Recording": list(tmp.index),
            "Duration (h:m:s)": hms,
            "Duration (s)": list(tmp["duration"]),
        }
    )
    print("Total audio recording")
    print(
        "Total recording of sound types: (h)",
        summary_total_recording["Duration (s)"].sum() / 3600,
    )
    print(
        summary_total_recording[["Recording", "Duration (h:m:s)"]].to_latex(index=False)
    )

    annot = ann.read_annotation_files(all_annot_files)
    annot = ann.apply_label_dict(annot, call_dict)
    annot["duration"] = annot["stop"] - annot["start"]
    total_call_duration = annot[["duration", "label"]].groupby(["label"]).sum()
    total_call_counts = annot[["duration", "label"]].groupby(["label"]).count()
    total_duration = total_call_duration["duration"].sum()
    dur = [aux.seconds_to_hms(x) for x in total_call_duration["duration"]]
    summary_calls = pd.DataFrame(
        {
            "Sound types": list(total_call_duration.index),
            "Total duration (h:m:s)": dur,
            "Total duration (s)": list(total_call_duration["duration"]),
            "Number": list(total_call_counts["duration"]),
        }
    )
    summary_calls = summary_calls[
        summary_calls["Sound types"].isin(calls_for_labeling_list)
    ]
    print(
        "Total recording of sound types: (h)",
        summary_calls["Total duration (s)"].sum() / 3600,
    )
    print(
        summary_calls[["Sound types", "Total duration (h:m:s)", "Number"]].to_latex(
            index=False
        )
    )
