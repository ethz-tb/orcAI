
# OrcAI

## Summary

OrcAI is a software package to detect acoustic signals in spectrograms generated from audio recordings. It is trained on audio recordings of killer whales producing a variety of intentional acoustic signals produced for communication (such as calls, whistles, herding calls, and buzzes) as well as sounds not intended for communication (such as prey handling sounds, breathing or tailslaps).

OrcAI uses audio recordings together with annotations of the above sound types to train machine learning models which can then be used to predict annotation of sounds patterns found in recordings that have not yet been annotated.

The package contains code to perform to distinct three sets of tasks:

- The first set concerns the production of data for training, validation and testing of the machine learning models from the raw audio files and accompanying annotations.
- The second set uses the generated training, validation and test data to develop and train models for prediction
- The third set is to apply these models to predict annotation in as of yet unannotated recordings and, in as far as this is required, to post-process the predicted annotations.

## Reference

orcAI has been published in the Journal Marine Mammal Science as:

Bonhoeffer, S. et al. 2025. “orcAI: A Machine Learning Tool to Detect and Classify Acoustic Signals of Killer Whales in Audio Recordings.” Marine Mammal Science 42 (1): e70083. https://doi.org/10.1111/mms.70083.

```bibtex
@article{https://doi.org/10.1111/mms.70083,
  author = {Bonhoeffer, Sebastian and Selbmann, Anna and Angst, Daniel C. and Ochsner, Nicolas and Miller, Patrick J. O. and Samarra, Filipa I. P. and Baumgartner, Chérine D.},
  title = {orcAI: A Machine Learning Tool to Detect and Classify Acoustic Signals of Killer Whales in Audio Recordings},
  journal = {Marine Mammal Science},
  volume = {42},
  number = {1},
  pages = {e70083},
  keywords = {bioacoustics, cetaceans, deep learning},
  doi = {https://doi.org/10.1111/mms.70083},
  url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/mms.70083},
  eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1111/mms.70083},
  note = {e70083 9425791},
  abstract = {ABSTRACT Acoustic monitoring is an essential tool for investigating animal communication and behavior when visual contact is limited, but the scalability of bioacoustic projects is often limited by time-intensive manual auditing of focal signals. To address this bottleneck, we introduce orcAI—a novel deep learning framework for the automated detection and classification of a broad acoustic repertoire of killer whales (Orcinus orca), including vocalizations (e.g., pulsed calls, whistles) and incidental sounds (e.g., breathing, tail slaps). orcAI combines a ResNet-based Convolutional Neural Network (ResNet-CNN) with Long Short-Term Memory (LSTM) layers to capture both spatial features and temporal context, enabling the model to classify signals and to accurately determine their temporal boundaries in spectrograms. Trained on a comprehensive dataset from herring-feeding killer whales off Iceland, the framework was designed to be adaptable to other populations upon training with equivalent data. Our final model achieves up to 98.2\% accuracy on test data and is delivered as an open-source tool with an easy-to-use command-line interface. By providing a ready-to-use model that processes raw audio and outputs annotations, orcAI serves as a useful tool for advancing the study of killer whale vocal behavior and, more broadly, for understanding marine mammal communication and ecology.}
}
```

## Installation


orcAI requires Python 3.11 and should work on all platforms supported by Python. It has been tested extensively on macOS (version 26.2) on Apple Silicon and Ubuntu (version 22.04.5 LTS). Cursory testing (install & prediction) has been performed on Win10 (Build 17763.8146) and Win11 (Build 26100.7171).

orcAI can be installed using standard python tools such as `pip`. To install as tool for command line usage you can use e.g. [uv](https://docs.astral.sh/uv/).
After installing uv, you can install the latest version of orcAI by running the following command:

```bash
uv tool install git+https://github.com/ethz-tb/orcAI.git --python 3.11
```

To updgrade an existing installation, run the following command:

```bash
uv tool upgrade orcai
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

An appropriate recording table can be created using the `orcai create-recording-table` command.

See `orcai predict --help` for more options.

### `orcai filter-predictions`

Example usage:

```bash
orcai filter-predictions path/to/annotations.txt
```

This will filter the annotations in the input file `path/to/annotations.txt` based on the minimum and maximum duration specified in the default configuration file. The output will be saved in the same directory as the input file with the same name but with the extension `_filtered.txt`. To pass a custom configuration file, use the `--call_duration_limits` option.
See `orcai filter-predictions --help` for more options.

## Usage for data preperation and training

All commands are documented, use `orcai command --help`.
For a full example of the usage please see the code accompanying the manuscript: Please see [orcai-ms-code](https://github.com/ethz-tb/orcai-ms-code).
