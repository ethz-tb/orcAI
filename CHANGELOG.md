# Changelog

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


