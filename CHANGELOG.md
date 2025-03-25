# Changelog

## [0.7.0] - unreleased

### Changes

- set validation steps in model.fit

### Added

- added hyperparameter search cli

## [0.6.1] - 2025-03-24

### Changes

- extract one additional batch to try to avoid the warning `/keras/src/trainers/epoch_iterator.py:151: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least 'steps_per_epoch * epochs' batches. You may need to use the '.repeat()' function when building your dataset.`

## [0.6.0] - 2025-03-24

### Changes

- **Breaking** revert to using TFRecordWriter for saving datasets. The warning `/keras/src/trainers/epoch_iterator.py:151: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least 'steps_per_epoch * epochs' batches. You may need to use the '.repeat()' function when building your dataset.` probably is not relevant?

## [0.5.0] - 2025-03-24

### Changes

- **Breaking** save datasets using tf.data.Datset.save instead of TFRecordWriter in an effort to get rid of `/keras/src/trainers/epoch_iterator.py:151: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least 'steps_per_epoch * epochs' batches. You may need to use the '.repeat()' function when building your dataset.` warning.

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

- **Breaking**: Combined default_calls.json, default_model_parameter.json, default_spectrogram_parameter.json, default_snippet_parameter.json into a single default_orcai_parameter.json file.
- **Breaking**: new DataLoader
- **Breaking**: new save format for data: TFRecord
- **Breaking**: renamed module `load` to `io`, rename 'reload_dataset' to 'load_dataset'
- **Breaking**: renamed module `annotations` to `labels`
- **Breaking**: moved io function from `auxiliary` to `io`
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

- **Breaking**: rename model architectures to correspond to manuscript (`efd3472c`)
- **Breaking**: remove multiprocessing from predict. It wasn't working and caused more problems than it solved. -np parameter is not available anymore.
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

_First prerelease._


[0.1.0]:https://gitlab.ethz.ch/tb/orcai/-/tags/v0.1.0
[0.2.0]:https://gitlab.ethz.ch/tb/orcai/-/tags/v0.2.0
[0.2.1]:https://gitlab.ethz.ch/tb/orcai/-/tags/v0.2.1
[0.3.0]:https://gitlab.ethz.ch/tb/orcai/-/tags/v0.3.0
[0.4.0]:https://gitlab.ethz.ch/tb/orcai/-/tags/v0.4.0
[0.4.1]:https://gitlab.ethz.ch/tb/orcai/-/tags/v0.4.1
[0.5.0]:https://gitlab.ethz.ch/tb/orcai/-/tags/v0.5.0
[0.6.0]:https://gitlab.ethz.ch/tb/orcai/-/tags/v0.6.0
[0.6.1]:https://gitlab.ethz.ch/tb/orcai/-/tags/v0.6.1


