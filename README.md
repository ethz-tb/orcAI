
# OrcAI

**THIS IS A WORK IN PROGRESS. ALL FUNCTIONALITY IS SUBJECT TO CHANGE**

## Summary

OrcAI is a software package to detect acoustic signals in spectrograms generated from audio recordings. It is trained on audio recordings of killer whales producing a variety of intentional acoustic signals produced for communication (such as calls, whistles, herding calls, and buzzes) as well as sounds not intended for communication (such as prey handling sounds, breathing or tailslaps).

OrcAI uses audio recordings together with annotations of the above sound types to train machine learning models which can then be used to predict annotation of sounds patterns found in recordings that have not yet been annotated.

The package contains code to perform to distinct two distinct sets of tasks. The first set of task is to produce data for training, validation and testing of the machine learning models from the raw audio files and accompanying annotations. The second set of tasks it to use the generated training, validation and test data to develop and train models for prediction and apply these models to predict annotation in as of yet unannotated recordings.

## Installation

orcAI can be installed using pip:

```bash
pip install -U git+https://gitlab.ethz.ch/tb/orcai.git
```

## Command Line Interface

The command line interface is available through the `orcai` and subcommands. The following subcommands are available:

- `orcai predict` - Predict annotations in unannotated recordings based on a trained model. A trained model is included in the package.
- `orcai filter-annotations` - Filter annotations based on minimum and maximum duration

### Predict

Example usage:

```bash
orcai predict path/to/input.wav
```

This will use the included model `orcai-V1` to predict annotations in the input file `path/to/input.wav`. The output will be saved in the same directory as the input file with the same name but with the extension `_orcai-V1_predicted.txt` and is compatible with Audacity.
See `orcai predict --help` for more options.

### Filter predictions

Example usage:

```bash
orcai filter-predictions path/to/annotations.txt
```

This will filter the annotations in the input file `path/to/annotations.txt` based on the minimum and maximum duration specified in the default configuration file. The output will be saved in the same directory as the input file with the same name but with the extension `_filtered.txt`. To pass a custom configuration file, use the `--call_duration_limits` option.
See `orcai filter-predictions --help` for more options.
