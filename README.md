
# OrcAI

## Summary

OrcAI is a software package to detect acoustic signals in spectrograms generated from audio recordings. It is trained on audio recordings of killer whales producing a variety of intentional acoustic signals produced for communication (such as calls, whistles, herding calls, and buzzes) as well as sounds not intended for communication (such as prey handling sounds, breathing or tailslaps).

OrcAI uses audio recordings together with annotations of the above sound types to train machine learning models which can then be used to predict annotation of sounds patterns found in recordings that have not yet been annotated.

The package contains code to perform to distinct two distinct sets of tasks. The first set of task is to produce data for training, validation and testing of the machine learning models from the raw audio files and accompanying annotations. The second set of tasks it to use the generated training, validation and test data to develop and train models for prediction and apply these models to predict annotation in as of yet unannotated recordings.

## Create data

The programs need to be executed in this order

- create_spectrogram.py: Generates spectrograms from wav files
- create_labels.py: Generates labels from corresponding annotation files
- create_snippets.py: Create list of paired snippets of spectrograms and labels
- create_tvt_data: Uses the list of paired snippets of spectrograms and labels to generate training, validation and test data sets for training

# Train models

- train_model.py: trains (depending on input) different types of models (cnn_res, dnn_res_lstm, or cnn_res_transformer) based on parameters provided
- hyperparameter_search.py: Performs a hyperparameter search for a specified model with given sets of parameters
- test_model.py: Runs model on test data
