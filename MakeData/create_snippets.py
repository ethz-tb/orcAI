#!/usr/bin/env python

# # %%
# extract from specs for train/val/test

# %%
# import
import numpy as np
import pandas as pd
from pathlib import Path
import zarr
import os
import sys


# import local
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import spectrogram as spec
import auxiliary as aux
import annotation as ann

# import load as load


# %%
# Read command line if interactive
interactive = aux.check_interactive()
if not interactive:
    print("Command-line call:", " ".join(sys.argv))
    computer, project_dir, model_name, mode = aux.create_snippets_commandline_parse()
else:
    computer = "laptop"
    model_name = "cnn_res_lstm_model"
    project_dir = "/Users/sb/polybox/Documents/Research/Sebastian/OrcAI_project/"
    mode = "read_in_existing_snippets"


# %%
# Read parameters
print("Project directory:", project_dir)
os.chdir(project_dir)

print("READ IN PARAMETERS")
dicts = {
    "directories_dict": "GenericParameters/directories.dict",
    "call_dict": "GenericParameters/call.dict",
    "spectrogram_dict": "GenericParameters/spectrogram.dict",
    "segments_dict": "GenericParameters/segments.dict",
    "model_dict": project_dir + "Results/" + model_name + "/model.dict",
    "calls_for_labeling_list": "GenericParameters/calls_for_labeling.list",
    "extract_snippets_dict": "GenericParameters/extract_snippets.dict",
}
for key, value in dicts.items():
    print("  - reading", key)
    globals()[key] = aux.read_json(value, True)


# %%
# Functions to extract snippets
def list_extract_snippets(
    fnstem, n_segments, segments_dict, directories_dict, model_dict
):
    """generates times for snippets to be extracted from fnstem
    returns pandas dataframe with fnstem, type, start, stop and duration for each label names
    """

    fnlabel = (
        directories_dict[computer]["root_dir_spectrograms"]
        + fnstem
        + "/labels/zarr.lbl"
    )
    try:
        label_filepointer = zarr.open(fnlabel, mode="r")
        fnlabel_list = (
            directories_dict[computer]["root_dir_spectrograms"]
            + fnstem
            + "/labels/label_list.json"
        )
        label_list_dict = aux.read_json(fnlabel_list, print_out=False)
        label_names = list(label_list_dict.keys())
        fn_times = (
            directories_dict[computer]["root_dir_spectrograms"]
            + fnstem
            + "/spectrogram/times.json"
        )
        t_vector = aux.read_json_to_vector(fn_times)
        delta_t = t_vector[1] - t_vector[0]
        n_filters = len(model_dict["filters"])
        i_duration = int(
            (2**n_filters) * ((segments_dict["duration"] / delta_t) // (2**n_filters))
        )  # to make time axis divisible by 2 ** n_filters
        print(" - i_duration", i_duration)
        label_duration_snippets = []
        for i_segment in range(n_segments):  # iterate over all segments
            print(" - segment", i_segment + 1, "of", n_segments)
            slice = (0, 0)
            for type in list(["train", "val", "test"]):  # iterate over type of snippet
                slice = (slice[1], slice[1] + model_dict[type])
                t_min = (i_segment + slice[0]) * segments_dict["length"]
                for j in range(
                    int(
                        model_dict[type]
                        * segments_dict["length"]
                        * segments_dict["per_sec"]
                    )
                ):  # iterate over number of snippets per segment and type
                    t_max = (i_segment + slice[1]) * segments_dict[
                        "length"
                    ] - segments_dict["duration"]
                    t_start = np.random.uniform(low=t_min, high=t_max, size=1)[0]

                    # Find the max index where entries are smaller than t_start
                    index_t_start = np.searchsorted(t_vector, t_start, side="left") - 1
                    # Find the min index where entries are smaller or equal to t_stop
                    index_t_stop = index_t_start + i_duration
                    label_chunk = label_filepointer[index_t_start:index_t_stop, :]
                    label_duration_snippet = label_chunk.sum(axis=0) * delta_t
                    label_duration_snippet[label_duration_snippet < 0] = np.nan
                    label_duration_snippets += [
                        list([fnstem, type, index_t_start, index_t_stop])
                        + list(label_duration_snippet)
                    ]
        return pd.DataFrame(
            label_duration_snippets,
            columns=["fnstem", "type", "row_start", "row_stop"] + label_names,
        )
    except:
        print("WARNING: cannot open label zarr file linked to", fnstem)


def compute_snippet_stats(es):
    """input: pandas dataframe with extracted snippets"""
    total_seconds_per_call_from_snippets = es[calls_for_labeling_list].sum()
    equalizing_factor = (
        1
        / total_seconds_per_call_from_snippets
        * total_seconds_per_call_from_snippets.max()
    )
    snippet_stats = pd.DataFrame(total_seconds_per_call_from_snippets)
    snippet_stats.rename(columns={0: "total(s)"}, inplace=True)
    snippet_stats["equalizing_factor"] = equalizing_factor
    return snippet_stats


def filter_no_label_snippets(es, fraction_removal, indices_no_label):
    """
    input:
        es: extracted_snippets (pandas dataframe)
        fraction_removal: fraction of no_label snippets to be removed
        indices_no_label: list of indices of extracted_snippets without any label
    return:
        extracted_snippets with corresponding fraction of no_label snippets removed
    """
    indices_to_drop = np.random.choice(
        indices_no_label,
        size=int(fraction_removal * len(indices_no_label)),
        replace=False,
    )
    indices_to_drop.sort()
    es = es.reset_index(drop=True)
    es = es[~es.index.isin(indices_to_drop)]
    indices_no_label = np.where(es[calls_for_labeling_list].sum(axis=1) <= 0.0000001)[0]
    print(
        "After selection: percentage of snippets containing no label:",
        np.around(100 * len(indices_no_label) / es.shape[0], 2),
        "%",
    )
    return es.reset_index(drop=True)


# %%
# reading in or generating new snippets
if mode == "read_in_existing_snippets":
    extracted_snippets = pd.read_csv(
        project_dir + "Results/" + "extracted_snippets.csv.gz"
    )

if mode == "generate_new_snippets":
    # get all valid wav and annotation files on the computer
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

    annotations = ann.read_annotation_files(all_annot_files)
    annotations = ann.apply_label_dict(annotations, call_dict)
    annotation_statistics = aux.calculate_total_duration(annotations)
    print("Annotation statistics:")
    print(annotation_statistics.to_string())

    all_annotated_wav_files = [x.replace(".txt", ".wav") for x in all_annot_files]
    fnstem_duration = aux.get_wav_duration(all_annotated_wav_files)
    total_recording_time = fnstem_duration["duration"].sum()
    print(
        "total duration of annotated wav files:",
        np.around(total_recording_time / 3600, 1),
        "h",
    )
    # add column n_segments to fnstem_duration
    fnstem_duration["n_segments"] = (
        fnstem_duration["duration"] // segments_dict["length"]
    ).astype(int)

# %%
# Generating new snippets
if mode == "generate_new_snippets":

    # generating snippets for all fnstem
    len_iter = len(fnstem_duration)
    for iter in range(len_iter):
        print(
            " extracting from fnstem:",
            fnstem_duration.iloc[iter]["fnstem"],
            "(",
            iter,
            "of",
            len_iter,
            ")",
        )
        if fnstem_duration.iloc[iter]["n_segments"] > 0:
            es = list_extract_snippets(
                fnstem_duration.iloc[iter]["fnstem"],
                fnstem_duration.iloc[iter]["n_segments"],
                segments_dict,
                directories_dict,
                model_dict,
            )
            first_es_created = True
        else:
            print(
                "duration (",
                fnstem_duration.iloc[iter]["duration"],
                ") shorter than segment length (",
                segments_dict["length"],
                ") in fnstem (",
                fnstem_duration.iloc[iter]["fnstem"],
                ")",
            )
        if iter == 0 and first_es_created:
            extracted_snippets = es
        else:
            extracted_snippets = pd.concat([extracted_snippets, es])
    extracted_snippets.reset_index()

    print("These fnstems are not used in extracted_snippets")
    print(set(fnstem_duration["fnstem"]) - set(extracted_snippets["fnstem"].unique()))

    # write extracted_snippets to disk
    extracted_snippets.to_csv(
        project_dir + "Results/" "extracted_snippets.csv.gz",
        compression="gzip",
        index=False,
    )


# %%
# statistics

print("Statistics snippets: all extracted snippets")
snippet_stats = {}
for type in ["train", "val", "test"]:
    snippet_stats[type] = compute_snippet_stats(
        extracted_snippets[extracted_snippets["type"] == type]
    )["total(s)"]
index = list(pd.DataFrame(snippet_stats).index)
for key, value in snippet_stats.items():
    ll = []
    for l in value:
        ll += [aux.seconds_to_hms(l)]
    snippet_stats[key] = ll
snippet_stats = pd.DataFrame(snippet_stats)
snippet_stats.index = index
print(pd.DataFrame(snippet_stats).to_latex())
indices_no_label = np.where(
    extracted_snippets[calls_for_labeling_list].sum(axis=1) <= 0.0000001
)[0]
print(
    "Before selection: percentage of snippets containing no label:",
    np.around(100 * len(indices_no_label) / extracted_snippets.shape[0], 2),
    "%",
)
# removing rows from extracted_snippets where there is no label present
fraction_removal = extract_snippets_dict["fraction_removal"]
print("  - removing ", np.around(fraction_removal * 100, 2), "% of no_label")
es = filter_no_label_snippets(extracted_snippets, fraction_removal, indices_no_label)
print(
    "number of train, val, test, snippets:",
    len(es[es["type"] == "train"]),
    len(es[es["type"] == "val"]),
    len(es[es["type"] == "test"]),
)


# %%
# extract and save train_df, val_df and test_df
def extract_data_set(es, type, n, directories_dict):
    """extracts from es['type'==type] randomly n rows"""
    root_dir_spectrograms = directories_dict[computer]["root_dir_spectrograms"]
    root_dir_tvtdata = directories_dict[computer]["root_dir_tvtdata"]
    df = es[es["type"] == type].sample(n=n, replace=False)
    fnstem_path = [root_dir_spectrograms + x + "/" for x in list(df["fnstem"])]
    df["fnstem_path"] = fnstem_path
    if not os.path.exists(root_dir_tvtdata):
        os.makedirs(root_dir_tvtdata)
    fn_name = root_dir_tvtdata + type + ".csv.gz"
    df = df[["fnstem_path", "row_start", "row_stop"]]
    print("saving", n, type, "snippets to ", fn_name)
    df.to_csv(fn_name, compression="gzip", index=False)
    return df


# generating train_df
if len(es[es["type"] == "train"]) < extract_snippets_dict["n_train"]:
    n = len(es[es["type"] == "train"])
    print('WARNING: extract_snippets_dict["n_train"] larger than available snippets')
else:
    n = extract_snippets_dict["n_train"]
train_df = extract_data_set(es, "train", n, directories_dict)

# generating val_df
if len(es[es["type"] == "val"]) < extract_snippets_dict["n_val"]:
    n = len(es[es["type"] == "val"])
    print('WARNING: extract_snippets_dict["n_val"] larger than available snippets')
else:
    n = extract_snippets_dict["n_val"]
val_df = extract_data_set(es, "val", n, directories_dict)

# generating test_df
if len(es[es["type"] == "test"]) < extract_snippets_dict["n_test"]:
    n = len(es[es["type"] == "test"])
    print('WARNING: extract_snippets_dict["n_test"] larger than available snippets')
else:
    n = extract_snippets_dict["n_test"]
test_df = extract_data_set(es, "test", n, directories_dict)


# %%
print("PROGRAM COMPLETED")

# %%
