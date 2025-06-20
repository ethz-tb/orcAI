# Changelog

## [1.0.3] - 2025-06-13

- more refactoring for GUI project
- fix for filtering of predictions (unfortunately in same commit.)


## [1.0.2] - 2025-06-11

### Changes

- refactor for GUI project (https://github.com/ethz-tb/orcai-gui)


## [1.0.1] - 2025-05-09

### Changes

- change column order of confusion table
- switch development backend to uv (https://docs.astral.sh/uv/)

## [1.0.0] - 2025-05-07

### Changes

- v1.0.0
- move orcai-v1.md to seperate repository (https://github.com/ethz-tb/orcai-ms-code)

## [0.25.0] - 2025-05-06

### Changes

- remove tensorflow-metal from dependencies. With the metal plugin the model
  training & prediction is much worse than on Euler with CUDA. Reason unknown.
  - __ATTENTION__: Please refresh your environment or uninstall tensorflow-metal
- update orcai-v1 pipeline with tests


## [0.24.0] - 2025-05-05

### Changes

- update trained orcai-v1 model
- create unfiltered test set when creating tvt data
- __Breaking__: update test.py to use new unfiltered test set

## [0.23.3] - 2025-05-05

### Changes

- add overwrite flag to predict
- better logging.

## [0.23.2] - 2025-05-05

### Changes

- general polish

## [0.23.1] - 2025-05-05

### Changes

- update default parameters for orcai-v1
- update default parameters for hyperparameter search
- move remote to github.

## [0.23.0] - 2025-05-02

### Changes

- subclass layer for frequency reduction in 1DConv model for serialization. Fixes serialization of 1D Conv models.
- save data of all trials as csv in hpsearch

## [0.22.1] - 2025-05-01

### Changes

- Buffer size back to 1000. OOM errors on euler with 20000.

## [0.22.0] - 2025-04-30

### Changes

- __Breaking__: rename module hp_search to hpsearch
- Potential fix for training routine that lead to incomplete epochs because not enough batches could be extracted.
- increase Shuffle buffer size to 20000
- add initializers as parameters to model architectures.
  - two new parameter in orcai_parameter["model"]
    - "conv_initializer": default "glorot_uniform" (the keras default), "he_normal" might be better suited for layers with ReLU activations
    - "lstm_initializer": default "glorot_uniform" (the keras default), which should be fine for LSTMs. Included for completeness.
    - if values are not in orcai_parameters.json, the default values are used.
- __Breaking__: removed maskedAUC reporting. Class is still available but unused.
  - parameter orcai_parameter["monitor"] can only be "val_MBA"
- minor refactoring

## [0.21.1] - 2025-04-29

### Changes

- fix ResNet1DConv for keras 3

## [0.21.0] - 2025-04-29

### Changes

- __Breaking__: switch back to keras 3. keras_tuner is not compatible with tf_keras.

## [0.20.1] - 2025-04-28

### Changes

- fix path passed to to callbacks

## [0.20.0] - 2025-04-28

### Changes

- __Breaking__: remove `--save_best_model` flag from `hpsearch` command
- update pipeline.
- minor prettifications.

## [0.19.0] - 2025-04-28

### Changes

- __Breaking__: filter duplicate snippets. These duplicates stem from randomly sampling.
- Minor reformatting of code and help


## [0.18.0] - 2025-04-24

### Changes

- __Breaking__: downgrade keras to keras 2 (tf-keras)
- __Breaking__: move JSONEncoderExt to seperate module to optimize loading
- __Breaking__: new parameter in orcai_parameter: "ReduceLROnPlateau_min_learning_rate": 1e-06
  - add to orcai_parameters.json to unbreak.

### Notes

- keras 2 trains better models. Reason unknown.
- tried to revert to Sebs DataLoader function, inheriting from keras.utils.Sequence. No difference detected except worse performance.


## [0.17.0] - 2025-04-22

### Changes

- cleanup imports
- refactor error-prone sorting of annotations
- __Breaking__: new parameter in orcai_parameter: "EarlyStopping_patience": 10, "ReduceLROnPlateau_patience": 3, "ReduceLROnPlateau_factor": 0.5
  - add to orcai_parameters.json to unbreak.

## [0.16.0] - 2025-04-22

### Changes

- __Breaking__: in train, load model instead of weights when restarting
  - the argument `load_weights` is now `load_model`
  - the flag option in the cli `--load_weights` is now `--load_model`
- __Breaking__: implement class weights.
  - orcai_parameters.json now has a model.call_weights key.
    - to unbreak add `"call_weights": null` to the model section of orcai_parameters.json
  - Implemented "three" methods for calculating call weights:
    - `"balanced"` is the same heuristic as is used in sklearn (= total / (n_calls \* count)),
    - `"max"` is 1/count \* total,
    - `"uniform"` is all ones and equal to None.
    - Use `null` (in Json, `None` in python) to disable class weights.
- __Breaking__: switch to class based metrics
- __Breaking__: implement AUC ROC metric
- __Breaking__: making the choice of metric to monitor for callbacks an option in orcai_parameters.json
  - to unbreak add `"monitor": "val_MBA"` to the model section of orcai_parameters.json
- make ReduceLROnPlateau callback patience == model_parameters['patience'] // __3__
- define metrics in architectures.py
- new arg parameter to overwrite default orcai parameter on project init

## [0.15.1] - 2025-04-15

### Changes

- make ReduceLROnPlateau callback patience == model_parameters['patience'] // 2

## [0.15.0] - 2025-04-14

### Changes

- save best hyperparameters
- flag to save model when running hp_search

## [0.14.0] - 2025-04-11

### Changes

- __Breaking__ save all_snippets.csv.gz in tvt_data not recording data
- enable changing of initial learning rate with orcai_parameters.json
- switch formatter and format imports

## [0.13.2] - 2025-04-11

### Changes

- fix: set backend explicitly


## [0.13.1] - 2025-04-11

### Changes

- set backend explicitly
- use class for loss function and set from_logits to False (should be equivalent)
- use `__version__` attribute for version number


## [0.13.0] - 2025-04-08

### Changes

- update zarr to zarr>=3,<4. Is backwards compatible in terms of loading files


## [0.12.2] - 2025-04-08

### Changes

- fix for zarr 2.8.15: fix numcodecs to < 0.16 (until zarr is updated or we switch to zarr 3.0)

## [0.12.1] - 2025-04-08

### Changes

- fix dataset compression option in cli
- fallback dataset_shape
- fix keras imports

## [0.12.0] - 2025-04-07

### Changes

- add option for dataset compression to create_tvt_data, train, test and hp_search
- save test results to subdir by default
- improve saved tables in test

## [0.11.1] - 2025-04-03

### Added

- report system information in train

## [0.11.0] - 2025-04-03

### Changes

- __Breaking__ save dataset using tf.data.Dataset.save instead of TFRecordWriter
- move loading model to fn in orcAI.io
- move saving model to fn in orcAI.io


## [0.10.0] - 2025-03-28

### Changes

- __Breaking__ unify seed generation (will break reproducibility). The seeds
  used take care of the random number generation for sampling and shuffling.
  However, there are certain non-deterministic behaviours in python and the CUDA runtime
  anyway so reproducibility is not guaranteed in any case.
- Shuffle indices in DataLoader. Originally removed because dataset is shuffled anyway after loading.
- fix data loading in test
- more docstrings and type hints

### Added

- add LICENCE file (in addition to specifying it in pyproject.toml)

## [0.9.1] - 2025-03-27

### Changes

- fix default for predict
- update readme

## [0.9.0] - 2025-03-27

### Changes

- __Breaking__ move hp_search to seperate module
- fix predict function stopping if a single prediction fails
- fix saving project name in `init` command

### Added

- docstrings for predict functions
- add type hints to save functions

## [0.8.1] - 2025-03-25

### Changes

- fix passing dataset size to load_dataset

## [0.8.0] - 2025-03-25

### Changes

- set cardinality when loading datasets
- restructure predict fn
  - load model only once
  - remove redundant predict_wav function
  - load model from .keras for newer models
- update predict cli
  - option to choose from included models (now includes orcai-v1 and orcai-v1-6400 trained on 6400 batches)

## [0.7.0] - 2025-03-25

### Changes

- explicitly set validation steps in model.fit
- report and save duration by call of selected snippets when creating tvt snippet tables

### Added

- added hyperparameter search cli
- report and save duration of selected snippets

## [0.6.1] - 2025-03-24

### Changes

- extract one additional batch to try to avoid the warning `/keras/src/trainers/epoch_iterator.py:151: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least 'steps_per_epoch * epochs' batches. You may need to use the '.repeat()' function when building your dataset.`

## [0.6.0] - 2025-03-24

### Changes

- __Breaking__ revert to using TFRecordWriter for saving datasets. The warning `/keras/src/trainers/epoch_iterator.py:151: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least 'steps_per_epoch * epochs' batches. You may need to use the '.repeat()' function when building your dataset.` probably is not relevant?

## [0.5.0] - 2025-03-24

### Changes

- __Breaking__ save datasets using tf.data.Datset.save instead of TFRecordWriter in an effort to get rid of `/keras/src/trainers/epoch_iterator.py:151: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least 'steps_per_epoch * epochs' batches. You may need to use the '.repeat()' function when building your dataset.` warning.

### Added

- addded test cli



## [0.4.2] - 2025-03-21

### Changed

- updated pyproject toml to include cuda and tensorrt for linux and windows
- minor fixes to logging

## [0.4.1] - 2025-03-21

### Changed

- cleaned up messaging

### Added

- Messenger now prints total duration and duration of the last parts for parts and success


## [0.4.0] - 2025-03-20

### Changed

- __Breaking__: Combined default_calls.json, default_model_parameter.json, default_spectrogram_parameter.json, default_snippet_parameter.json into a single default_orcai_parameter.json file.
- __Breaking__: new DataLoader
- __Breaking__: new save format for data: TFRecord
- __Breaking__: renamed module `load` to `io`, rename 'reload_dataset' to 'load_dataset'
- __Breaking__: renamed module `annotations` to `labels`
- __Breaking__: moved io function from `auxiliary` to `io`
- Channel added to the prediction output filename


### Added

- added function to initialize a project, copying the default parameter files
- added option to `predict' to save predicted probabilities
- added cli for making spectrograms
- added cli for creating label arrays
- added cli for creating snippet table
- added cli for create-tvt-snippet-tables
- added cli for create-tvt-data
- added cli for train
- added example pipeline

### Known Issues

- module `test` untested.

## [0.3.0] - 2025-03-14

### Changed

- __Breaking__: rename model architectures to correspond to manuscript (`efd3472c`)
- __Breaking__: remove multiprocessing from predict. It wasn't working and caused more problems than it solved. -np parameter is not available anymore.
- refactored predict_wav so that model is only loaded once in case of multiple predictions
- changed confusion matrix to confusion table reporting TP, FN, FP, TN, precision, recall, F1 score and number of samples

### Added

- docstrings in architectures.py
- restructure hyperparameter_search.py and move to train.py
- restructure test.py
- removed all tests from train.py

### Removes

- unused transformer models (`57794fed`)
- plot fn

## [0.2.1] - 2025-03-07

### Changed

- defer loading of cli commands to increase performance if only calling help. This massively speeds up the cli if only calling `--help` or `--version` (`f37fea01`)

## [0.2.0] - 2025-03-05

### Added

- `create_recordings_table`: Helper function for creating a recording table by scanning a directory
- added new `predict` function that allow passing of a recording table and annotating multiple recordings in parallel

### Changed

- renamed `predict` to `predict_wav`
- changed predict cli to accept a recording table
- `predict` now accepts a `call_duration_limits` file. If one is given predictions are filtered using `filter_predictions`
- cli now uses rich_click, mainly for grouping & sorting subcommands

## [0.1.0] - 2025-03-04

*First prerelease.*


[0.1.0]:https://github.com/ethz-tb/orcAI/releases/tag/v0.1.0
[0.2.0]:https://github.com/ethz-tb/orcAI/releases/tag/v0.2.0
[0.2.1]:https://github.com/ethz-tb/orcAI/releases/tag/v0.2.1
[0.3.0]:https://github.com/ethz-tb/orcAI/releases/tag/v0.3.0
[0.4.0]:https://github.com/ethz-tb/orcAI/releases/tag/v0.4.0
[0.4.1]:https://github.com/ethz-tb/orcAI/releases/tag/v0.4.1
[0.5.0]:https://github.com/ethz-tb/orcAI/releases/tag/v0.5.0
[0.6.0]:https://github.com/ethz-tb/orcAI/releases/tag/v0.6.0
[0.6.1]:https://github.com/ethz-tb/orcAI/releases/tag/v0.6.1
[0.7.0]:https://github.com/ethz-tb/orcAI/releases/tag/v0.7.0
[0.8.0]:https://github.com/ethz-tb/orcAI/releases/tag/v0.8.0
[0.8.1]:https://github.com/ethz-tb/orcAI/releases/tag/v0.8.1
[0.9.0]:https://github.com/ethz-tb/orcAI/releases/tag/v0.9.0
[0.9.1]:https://github.com/ethz-tb/orcAI/releases/tag/v0.9.0
[0.10.0]:https://github.com/ethz-tb/orcAI/releases/tag/v0.10.0
[0.11.0]:https://github.com/ethz-tb/orcAI/releases/tag/v0.11.0
[0.11.1]:https://github.com/ethz-tb/orcAI/releases/tag/v0.11.1
[0.12.0]:https://github.com/ethz-tb/orcAI/releases/tag/v0.12.0
[0.12.1]:https://github.com/ethz-tb/orcAI/releases/tag/v0.12.1
[0.12.2]:https://github.com/ethz-tb/orcAI/releases/tag/v0.12.2
[0.13.0]:https://github.com/ethz-tb/orcAI/releases/tag/v0.13.0
[0.13.1]:https://github.com/ethz-tb/orcAI/releases/tag/v0.13.1
[0.13.2]:https://github.com/ethz-tb/orcAI/releases/tag/v0.13.2
[0.14.0]:https://github.com/ethz-tb/orcAI/releases/tag/v0.14.0
[0.15.0]:https://github.com/ethz-tb/orcAI/releases/tag/v0.15.0
[0.15.1]:https://github.com/ethz-tb/orcAI/releases/tag/v0.15.1
[0.16.0]:https://github.com/ethz-tb/orcAI/releases/tag/v0.16.0
[0.17.0]:https://github.com/ethz-tb/orcAI/releases/tag/v0.17.0
[0.18.0]:https://github.com/ethz-tb/orcAI/releases/tag/v0.18.0
[0.19.0]:https://github.com/ethz-tb/orcAI/releases/tag/v0.19.0
[0.20.0]:https://github.com/ethz-tb/orcAI/releases/tag/v0.20.0
[0.20.1]:https://github.com/ethz-tb/orcAI/releases/tag/v0.20.1
[0.21.0]:https://github.com/ethz-tb/orcAI/releases/tag/v0.21.0
[0.21.1]:https://github.com/ethz-tb/orcAI/releases/tag/v0.21.1
[0.22.0]:https://github.com/ethz-tb/orcAI/releases/tag/v0.22.0
[0.22.1]:https://github.com/ethz-tb/orcAI/releases/tag/v0.22.1
[0.23.0]:https://github.com/ethz-tb/orcAI/releases/tag/v0.23.0
[0.23.1]:https://github.com/ethz-tb/orcAI/releases/tag/v0.23.1
[0.23.2]:https://github.com/ethz-tb/orcAI/releases/tag/v0.23.2
[0.23.3]:https://github.com/ethz-tb/orcAI/releases/tag/v0.23.3
[0.24.0]:https://github.com/ethz-tb/orcAI/releases/tag/v0.24.0
[0.25.0]:https://github.com/ethz-tb/orcAI/releases/tag/v0.25.0
[1.0.0]:https://github.com/ethz-tb/orcAI/releases/tag/v1.0.0
[1.0.1]:https://github.com/ethz-tb/orcAI/releases/tag/v1.0.1
[1.0.2]:https://github.com/ethz-tb/orcAI/releases/tag/v1.0.2
[1.0.3]:https://github.com/ethz-tb/orcAI/releases/tag/v1.0.3
