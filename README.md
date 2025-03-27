
# OrcAI

**THIS IS A WORK IN PROGRESS. ALL FUNCTIONALITY IS SUBJECT TO CHANGE**

## Summary

OrcAI is a software package to detect acoustic signals in spectrograms generated from audio recordings. It is trained on audio recordings of killer whales producing a variety of intentional acoustic signals produced for communication (such as calls, whistles, herding calls, and buzzes) as well as sounds not intended for communication (such as prey handling sounds, breathing or tailslaps).

OrcAI uses audio recordings together with annotations of the above sound types to train machine learning models which can then be used to predict annotation of sounds patterns found in recordings that have not yet been annotated.


OrcAI is a software package to detect acoustic signals in spectrograms generated from audio recordings. It is trained on audio recordings of killer whales producing a variety of intentional acoustic signals produced for communication (such as calls, whistles, herding calls, and buzzes) as well as sounds not intended for communication (such as prey handling sounds, breathing or tailslaps).

OrcAI uses audio recordings together with annotations of the above sound types to train machine learning models which can then be used to predict annotation of sounds patterns found in recordings that have not yet been annotated.

The package contains code to perform to distinct three sets of tasks:

- The first set concerns the production of data for training, validation and testing of the machine learning models from the raw audio files and accompanying annotations.
- The second set uses the generated training, validation and test data to develop and train models for prediction
- The third set is to apply these models to predict annotation in as of yet unannotated recordings and, in as far as this is required, to post-process the predicted annotations.



## Installation

orcAI can be installed using [pipx](https://pipx.pypa.io/stable/).
orcAI requires Python 3.11 and should work on all platforms supported by Python.

To install the latest version of orcAI, run the following command:

```bash
pipx install git+https://gitlab.ethz.ch/tb/orcai.git --python python3.11
```

## Command Line Interface

The command line interface is available through the `orcai` and subcommands. The following subcommands are available:

- Predicting calls
  - `orcai predict` - Predict annotations in unannotated recordings based on a trained model. A trained model is included in the package.
  - `orcai filter-annotations` - Filter annotations based on minimum and maximum duration
- Training models
  - `create-spectrograms`- Creates spectrograms.
  - `create-label-arrays`- Creates label arrays.
  - `create-snippet-table` - Creates snippet tables.
  - `create-tvt-snippet-tables` - Creates TVT snippet tables.
  - `create-tvt-data` - Creates TVT datasets.
  - `hpsearch` - Hyperparameter search.
  - `train` - Trains a model.
  - `test` - Tests a model.
- Helpers
  - `init` - Initializes a new orcAI project.
  - `orcai create-recording-table` - Create a recording table from a directory of recordings

## Usage for Prediction

### `orcai predict`

Basic usage:

```bash
orcai predict path/to/input.wav
```

This will use the included model `orcai-V1` to predict annotations in the input file `path/to/input.wav`. The output will be saved in the same directory as the input file with the same name but with the extension `_orcai-V1_predicted.txt` and is compatible with Audacity.

Advanced usage e.g. for predicting multiple recordings in parallel:

```bash
orcai predict path/to/recording_table.csv -o path/to/output_dir
```

This will use the included model `orcai-V1` to predict annotations in the recordings listed in the recording table `path/to/recording_table.csv`. The output will be saved in the directory `path/to/output_dir` with the same name as the input file but with the extension `_orcai-V1_predicted.txt` and is compatible with Audacity.

A appropriate recording table can be created using the `orcai create-recording-table` command, described below.

See `orcai predict --help` for more options.

### `orcai filter-predictions`

Example usage:

```bash
orcai filter-predictions path/to/annotations.txt
```

This will filter the annotations in the input file `path/to/annotations.txt` based on the minimum and maximum duration specified in the default configuration file. The output will be saved in the same directory as the input file with the same name but with the extension `_filtered.txt`. To pass a custom configuration file, use the `--call_duration_limits` option.
See `orcai filter-predictions --help` for more options.

## Usage for Data preperation and training

All commands are documented, use `orcai command --help`.
Please see [the example pipeline](orcai-v1.md) for a complete example of how to use orcAI to prepare data and train a model.
