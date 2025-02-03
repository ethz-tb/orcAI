# %%
# import
from pathlib import Path
import numpy as np
import numpy as np
import pandas as pd
import zarr


# import local
import auxiliary as aux


# %%
##########################
# FUNCTIONS
##########################


def read_annotation_file(fn):
    """read annotation file and return with fnstem as additional column"""
    try:
        df = pd.read_csv(
            fn,
            sep="\t",
            encoding="utf-8",
            header=None,
            names=["start", "stop", "origlabel"],
        )
        df["fnstem"] = Path(fn).stem
    except:
        print("WARNING: file not found in read_annotation_file:", fn)
        return
    return df[["fnstem", "start", "stop", "origlabel"]]


def read_annotation_files(fns):
    """read multiple annotation file and return with fnstem as additional column"""
    annot = pd.DataFrame()
    for fn in fns:
        a = read_annotation_file(fn)
        annot = pd.concat([annot, a])
    return annot


def apply_label_dict(df, dict):
    """map call_dict to origlabel"""
    df["label"] = df["origlabel"].map(dict)
    return df


def labelarr_from_annot(
    root_dir, annot_file, fnstem, call_dict, labels_present, labels_masked, mask_value
):
    """transform annot into array with 0 and 1 for pres/abs of each label at times t_vec"""

    try:
        read_annotation_file(annot_file)
    except:
        print("annotation file not found:", annot_file)
        return

    # get annotations
    fnstem = Path(annot_file).stem
    a = read_annotation_file(annot_file)
    a = apply_label_dict(a, call_dict)
    annotations = a[(a["fnstem"] == fnstem)][["start", "stop", "label"]]

    # load t_vec of spectrogram
    try:
        t_vec = aux.read_json_to_vector(root_dir + fnstem + "/spectrogram/times.json")
    except:
        print(" file not found: ", root_dir + fnstem + "/spectrogram/times.json")
        return

    # Initialize df with label_arr
    df = pd.DataFrame({})
    # Create a column for each label present
    for label in labels_present:
        # Find all intervals for the current label
        label_intervals = annotations[annotations["label"] == label]

        # Create a boolean mask for the current label
        bool_mask = np.zeros(len(t_vec), dtype=bool)

        # Check if each time step in t_vec is within any interval
        for start, stop in zip(label_intervals["start"], label_intervals["stop"]):
            bool_mask |= (t_vec >= start) & (t_vec <= stop)

        # Add the mask to the result DataFrame as a binary column
        df[label] = bool_mask.astype(int)

    # Create a column for each label masked
    for label in labels_masked:
        df[label] = mask_value * np.ones(
            len(t_vec), dtype=int
        )  # set mask value to -1 for label to be masked, set to zero if labels should be assumed absent

    # sort columns alphabetically
    sorted_columns = sorted(df.columns, key=str)
    df = df[sorted_columns]
    label_list = {}
    for lab in labels_present:
        label_list[lab] = "present"
    for lab in labels_masked:
        label_list[lab] = "masked"
    label_list = {key: label_list[key] for key in sorted(label_list)}

    # save to numpy
    label_arr = df.to_numpy()
    stem_dir = Path(annot_file).stem + "/"
    print("  - saving label_arr to disk for:", stem_dir)
    sub_dir = "labels/"
    zarr_fn = "/zarr.lbl"
    zarr_file = zarr.open(
        root_dir + stem_dir + sub_dir + zarr_fn,
        mode="w",
        shape=label_arr.shape,
        chunks=(2000, label_arr.shape[1]),
        dtype="float32",
        compressor=zarr.Blosc(cname="zlib"),
    )
    zarr_file[:] = label_arr

    # save label array to zarr
    aux.write_dict(label_list, root_dir + stem_dir + sub_dir + "/label_list.json")
    return


# %%
def reshape_label_arr(arr, n_filters):
    dim1 = arr.shape[0] // n_filters
    dim2 = arr.shape[1]
    if arr.shape[0] % n_filters == 0:
        arr_out = (arr.reshape(dim1, n_filters, dim2).mean(axis=1) + 0.5).astype(int)
    return arr_out
