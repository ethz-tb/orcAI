#!/usr/bin/env python

# %%
# import
import time
from pathlib import Path
import tensorflow as tf
import sys
import os


# import local
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import auxiliary as aux
import load


# %%
# Read command line if interactive
aux.print_memory_usage()
interactive = aux.check_interactive()

if not interactive:
    computer, project_dir, model_name = aux.create_tvtdata_commandline_parse()
else:
    computer = "laptop"
    project_dir = "/Users/sb/polybox/Documents/Research/Sebastian/OrcAI_project/"
    model_name = "cnn_res_lstm_model"


# %%
# Read parameters
print("READ IN PARAMETERS")
print("Project directory:", project_dir)
os.chdir(project_dir)

print("READ IN PARAMETERS")
dicts = {
    "directories_dict": "GenericParameters/directories.dict",
    "model_dict": project_dir + "Results/" + model_name + "/model.dict",
}

for key, value in dicts.items():
    print("  - reading", key)
    globals()[key] = aux.read_dict(value, True)


print("Project directory:", project_dir)
os.chdir(project_dir)


# %%
# read in train/val/test dataframes with snippets
start_time = time.time()
csv_list = ["train.csv.gz", "val.csv.gz", "test.csv.gz"]
print("Reading in dataframes with snippets and generating loader_list")
loader_list, spectrogram_chunk_shape, label_chunk_shape = load.load_data_from_csv(
    csv_list,
    model_dict,
    directories_dict,
    computer,
)

train_loader, val_loader, test_loader = loader_list
del loader_list
print("  - Spectrogram chunk shape:", spectrogram_chunk_shape)
print("  - Original label chunk shape:", label_chunk_shape)
print(f"  - Processing time: {time.time() - start_time:.2f} seconds")
aux.print_memory_usage()


# %%
# Shape and dtype of input data
start_time = time.time()
spectrogram_batch, label_batch = train_loader[0]
print("Data characteristics:")
print(f"  - data loading time per batch: {time.time() - start_time:.2f} seconds")
print("  - Input spectrogram batch shape:", spectrogram_batch.shape)
print("  - Input label batch shape:", label_batch.shape)  # Should match num_labels


# %%
# Creating tensorflow data sets
print("Creating  test, validation and training  datasets:")
start_time = time.time()
# Create datasets for tensorflow mode;
train_dataset = tf.data.Dataset.from_generator(
    lambda: load.data_generator(train_loader),
    output_signature=(
        tf.TensorSpec(
            shape=(spectrogram_batch.shape[1], spectrogram_batch.shape[2], 1),
            dtype=tf.float32,
        ),  # Single spectrogram shape
        tf.TensorSpec(
            shape=(label_batch.shape[1], label_batch.shape[2]), dtype=tf.float32
        ),  # Single label shape
    ),
)
val_dataset = tf.data.Dataset.from_generator(
    lambda: load.data_generator(val_loader),
    output_signature=(
        tf.TensorSpec(
            shape=(spectrogram_batch.shape[1], spectrogram_batch.shape[2], 1),
            dtype=tf.float32,
        ),  # Single spectrogram shape
        tf.TensorSpec(
            shape=(label_batch.shape[1], label_batch.shape[2]), dtype=tf.float32
        ),  # Single label shape
    ),
)
test_dataset = tf.data.Dataset.from_generator(
    lambda: load.data_generator(test_loader),
    output_signature=(
        tf.TensorSpec(
            shape=(spectrogram_batch.shape[1], spectrogram_batch.shape[2], 1),
            dtype=tf.float32,
        ),  # Single spectrogram shape
        tf.TensorSpec(
            shape=(label_batch.shape[1], label_batch.shape[2]), dtype=tf.float32
        ),  # Single label shape
    ),
)
print(f"  - time to create datasets: {time.time() - start_time:.2f} seconds")

aux.print_memory_usage()


# %%
# define scratchdir and add trailing "/"
def ensure_trailing_slash(path):
    if not path.endswith("/"):
        path += "/"
    return path


scratch_dir = ensure_trailing_slash(directories_dict[computer]["root_dir_tvtdata"])

# %%
# save data sets to local  disk
print("Saving test, val and train datasets to disk:", scratch_dir)
start_time = time.time()
load.save_dataset(scratch_dir + "test_dataset", test_dataset)
print(f"  - time to save test_dataset: {time.time() - start_time:.2f} seconds")
start_time = time.time()
load.save_dataset(scratch_dir + "val_dataset", val_dataset)
print(f"  - time to save val_dataset: {time.time() - start_time:.2f} seconds")
start_time = time.time()
load.save_dataset(scratch_dir + "train_dataset", train_dataset)
print(f"  - time to save train_dataset: {time.time() - start_time:.2f} seconds")


# %%
# get size of tvt directory
print("Size scratch dir:")


def get_directory_size(directory):
    total_size = sum(
        f.stat().st_size for f in Path(directory).rglob("*") if f.is_file()
    )
    return total_size / (1024**3)


size_in_gb = get_directory_size(scratch_dir)
print(f"  - total size of {scratch_dir}: {size_in_gb:.2f} GB")
dataset_names = ["train_dataset", "val_dataset", "test_dataset"]
for f in dataset_names:
    size_in_gb = get_directory_size(scratch_dir + f)
    print(f"     - size of {f}: {size_in_gb:.2f} GB")

# %%
print("PROGRAM COMPLETED")
