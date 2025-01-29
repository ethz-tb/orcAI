#!/usr/bin/env python

# %%
# import
import time
import sys
import os


import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
import keras_tuner as kt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import local
import auxilliary as aux
import model as mod
import load


# %%
# Read command line if interactive


aux.print_memory_usage()
interactive = aux.check_interactive()

if not interactive:
    (
        computer,
        model_name,
        data_dir,
        project_dir,
    ) = aux.hyperparameter_search_commandline_parse()
else:
    model_name = "cnn_res_lstm_model"
    computer = "laptop"
    data_dir = "/Users/sb/AI_data/tvtdata/"
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
    "model_dict": project_dir + "Results/" + model_name + "/model.dict",
    "calls_for_labeling_list": "GenericParameters/calls_for_labeling.list",
    "hyperparameter_dict": project_dir
    + "Results/"
    + model_name
    + "/hyperparameter.dict",
}
for key, value in dicts.items():
    print("  - reading", key)
    globals()[key] = aux.read_dict(value, True)


# print
for root, dirs, files in os.walk(data_dir):
    for file in files:
        print(os.path.join(root, file))  # Prints full file path


print("PROGRAM COMPLETED")
