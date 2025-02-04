
OrcAI is a software package to detect acoustic signals in spectrograms generated from audio recordings. It is trained on audio recordings of killer whales producing a variety of intentional acoustic signals produced for communication (such as calls, whistles, herding calls, and buzzes) as well as sounds not intended for communication (such as prey handling sounds, breathing or tailslaps).

OrcAI uses audio recordings together with annotations of the above sound types to train machine learning models which can then be used to predict annotation of sounds patterns found in recordings that have not yet been annotated.

The package contains code to perform to distinct two distinct sets of tasks. The first set of task is to produce data from training, validation and testing of the machine learning models from the raw audio files and accompanying annotations. The second set of tasks it to use the generated training, validation and test data to develop and train models for prediction and apply these models to predict annotation in as of yet unannotated recordings.
