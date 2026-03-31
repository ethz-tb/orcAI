# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses a just to organise development commands. The dev commands can be found in `.justfile`

This project uses `pre-commit` to configure pre-commit hooks.

## Architecture

OrcAI detects and classifies killer whale vocalizations (7 call types: BR, BUZZ, HERDING, PHS, SS, TAILSLAP, WHISTLE) in audio recordings using a ResNet-CNN + LSTM model.

### Three subsystems

1. **Data Preparation** — raw `.wav` + Audacity annotation `.txt` files → Zarr spectrograms → Zarr label arrays → TFRecords datasets
2. **Model Training/Tuning** — hyperparameter search (Keras Tuner), training, evaluation
3. **Prediction/Inference** — apply trained models to unannotated recordings, filter predictions by call duration

### Data pipeline (CLI commands in order)


create-spectrograms → create-label-arrays → create-snippet-table
  → create-tvt-snippet-tables → create-tvt-data → train → test


Prediction uses: `predict` → `filter-predictions`

### Key modules

| Module             | Role                                                                  |
| ------------------ | --------------------------------------------------------------------- |
| `cli.py`           | All CLI commands via rich-click                                       |
| `spectrogram.py`   | `.wav` → power spectrograms (librosa STFT, 48kHz, 0–16kHz)            |
| `labels.py`        | Annotation files → Zarr label arrays (0=absent, 1=present, -1=masked) |
| `snippets.py`      | Snippet tables and train/val/test splits                              |
| `architectures.py` | ResNet-CNN + LSTM (`ResNetLSTM`), custom masked loss/metric layers    |
| `io.py`            | Zarr/TFRecords I/O, `DataLoader` class, model serialization           |
| `train.py`         | Training loop with early stopping and LR scheduling                   |
| `predict.py`       | Inference on new recordings                                           |
| `test_models.py`   | Confusion matrices, misclassification tables, metrics                 |
| `hpsearch.py`      | Keras Tuner hyperparameter search                                     |
| `auxiliary.py`     | `Messenger` (logging), `MASK_VALUE = -1.0`, seed constants            |

### Model architecture

- Input: 4D spectrograms (time × frequency × channels × 1)
- CNN: Separable Conv2D residual blocks with batch norm + max pooling
- LSTM: 128 units for temporal context
- Output: per-timestep binary classification per call type
- Loss: `MaskedBinaryCrossentropy` — skips frames where label == `MASK_VALUE` (-1.0)

### Data formats

- Spectrograms: Zarr (3D: time × frequency, metadata as JSON)
- Labels: Zarr (2D: time × call-types)
- Datasets: GZIP-compressed TFRecords
- Models: Keras saved format, with `model_shape.json` and `orcai_parameter.json` sidecar files

### Built-in models

Stored in `src/orcai/models/`:

- `orcai-isl-v1` (default): Iceland herring-feeding killer whales
- `orcai-nor-v1`: Norwegian population (fine-tuned from isl-v1)

Default configs are in `src/orcai/defaults/`.

## General Instructions

When editing or creating a file clearly add a docstring on the first lines with:

- a title
- a very short description, avoid flowery language.
- correct LLM model identifier used in this session and the date of edit or creation

here is an example if you create a file:

```python
"""Tests for orcai.architectures Module

Tests for neural network architectures, custom layers, loss functions, and metrics.

Created using: claude-haiku-4.5 on 2026-03-30
"""
```

if you later edit the file, add or update a second line at the end like `Created using: claude-haiku-4.5 on 2026-03-30`
