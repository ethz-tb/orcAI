# orcai-v1 pipeline

## init environment

```bash
pipx install git+https://gitlab.ethz.ch/tb/orcai.git --version python3.11
```

```console
  installed package orcai 0.3.0, Python 3.11.0
  These apps are now globally available
    - orcai
done! ✨ 🌟 ✨
```

```bash
orcai --version
```

```console
orcai, version 0.4.0
```


## Initialize project project

```bash
orcai init /path/to/orcai-v1 orcai-v1
cd /path/to/orcai-v1
```

```console
🐳 Initializing project
🐳 Creating project directory: /Volumes/4TB/orcai_project/orcai-v1-dca [0:00:02]
    Creating orcai-v1_orcai_parameter.json
    Creating orcai-v1_hps_parameter.json
    Creating orcai-v1_call_duration_limits.json
    Setting seed
🐳 Project initialized. [0:00:02, 𝚫 0:00:00]
```

## create/update recording table

create recording table from directory with recordings.
update original_recording_table (containing possibilities of calls entered by Chérine).
Calls to label are in orcai-v1_orcai_parameter.json.
Files to exclude are in orcai-v1_files_exclude.json.

```bash
orcai create-recording-table ../orca_recordings/Acoustics/ \
  -o orcai-v1_recording_table.csv \
  -ut original_recording_table.csv \
  -p orcai-v1_orcai_parameter.json \
  -ep orcai-v1_files_exclude.json \
  -up
```

```console
🐳 Creating recording table
🐳 Resolving file paths [0:00:03]
🐳 Filtering 1552 wav files... [0:00:03, 𝚫 0:00:00]
    Remaining files after filtering files that contain ._: 961
    Remaining files after filtering files that contain _ChB: 961
    Remaining files after filtering files that contain _Chb: 961
    Remaining files after filtering files that contain Movie: 961
    Remaining files after filtering files that contain Norway: 961
    Remaining files after filtering files that contain _acceleration: 961
    Remaining files after filtering files that contain _depthtemp: 961
    Remaining files after filtering files that contain _H.: 961
    Remaining files after filtering files that contain _orig: 961
    Remaining files after filtering files that contain _old: 961
🐳 Filtering 897 annotations files... [0:00:03, 𝚫 0:00:00]
    Remaining files after filtering files that contain ._: 502
    Remaining files after filtering files that contain _ChB: 501
    Remaining files after filtering files that contain _Chb: 500
    Remaining files after filtering files that contain Movie: 500
    Remaining files after filtering files that contain Norway: 500
    Remaining files after filtering files that contain _acceleration: 495
    Remaining files after filtering files that contain _depthtemp: 490
    Remaining files after filtering files that contain _H.: 463
    Remaining files after filtering files that contain _orig: 426
    Remaining files after filtering files that contain _old: 391
    ‼️ 4 annotations with missing recordings: {'oo21_202b007', 'oo21_184a006', 'oo21_184a007', 'oo21_189a011'}. These will be ignored.
🐳 Saving recording table to orcai-v1_recording_table.csv [0:00:03, 𝚫 0:00:00]
    Total recordings: 961
    Total recordins with annotations: 387
🐳 Recordings table created. [0:00:03, 𝚫 0:00:00]
```

## Make spectrograms

Create all spectrograms in recording_table and save to recording_data.
Use orcai-v1_orcai_parameter.json for spectrogram parameters.
Only for spectrograms with annotations (-en) and spectrograms with possible
annotations (-enp).

```bash
orcai create-spectrograms orcai-v1_recording_table.csv recording_data \
  -p orcai-v1_orcai_parameter.json 
```

```console
🐳 Creating spectrograms
🐳 Reading recordings table [0:00:04]
    Excluded 574 recordings because they are not annotated.
    Excluded recordings because they lack any possible annotations:
        ['2015-07-29c' '2015-12-07l' '2015-13-07b' '2015-13-07c' '2015-17-07c'
 '2015-17-07h' '2015-18-07a' '2015-18-07f' '2015-21-07g' '2015-25-07f'
 '2015-25-07i' '2015-25-07j' '2015_07_14b' '2016-05-07C' '2016-12-07C'
 '2016-13-07K' '2016-16-07I' '2016-20-071008' '2016-24-07T323'
 '2016-24-07T328' '2016-25-07T338' '2016-27-07T352' '2016-27-07T363'
 'oo09_200a043' 'oo09_209a012' 'oo14_048a012' 'oo21_175a004'
 'oo21_182a004' 'oo21_184a020' 'oo22_195a004' 'oo22_195a005'
 'oo22_195a006' 'oo22_195a007' 'oo22_195a008' 'oo22_195a009'
 'oo22_195a010' 'oo22_195a011' 'oo22_195a012' 'oo22_195a014'
 'oo22_195a015' 'oo22_228a003' 'oo22_228a005' 'oo23_181a102'
 'oo23_181a105' 'oo23_188a098']
    Excluded 341 recordings because they already have spectrograms.
🐳 Creating 1 spectrograms [0:00:04, 𝚫 0:00:00]
Making spectrograms: 100%|██████████████████████████████████| 1/1 [00:25<00:00, 25.62s/it]
```

Duration for all spectrograms: ~3h 55min (Apple M3 Pro, 18GB)

## create label arrays

Create all label arrasy for recordings in recording_table and save to
recording_data.
Use orcai-v1_orcai_parameter.json parameters.
Unify call labels according to orcai-v1_call_equivalences.json.

```bash
orcai create-label-arrays orcai-v1_recording_table.csv recording_data \
  -p orcai-v1_orcai_parameter.json \
  -ce orcai-v1_call_equivalences.json
```

```console
🐳 Making label arrays
    Missing annotation files for 574 recordings. Skipping these recordings.
    Skipping 341 recordings because they already have Labels.
Converting annotation files: 100%|████████████████████████| 46/46 [00:00<00:00, 291.88recording/s]
    ‼️ No valid labels present in ['2015-07-29c', '2015-12-07l', '2015-13-07b', '2015-13-07c', '2015-17-07c', '2015-17-07h', '2015-18-07a', '2015-18-07f', '2015-21-07g', '2015-25-07f', '2015-25-07i', '2015-25-07j', '2015_07_14b', '2016-05-07C', '2016-12-07C', '2016-13-07K', '2016-16-07I', '2016-20-071008', '2016-24-07T323', '2016-24-07T328', '2016-25-07T338', '2016-27-07T352', '2016-27-07T363', 'oo09_200a043', 'oo09_209a012', 'oo14_048a012', 'oo21_175a004', 'oo21_182a004', 'oo21_184a020', 'oo22_195a004', 'oo22_195a005', 'oo22_195a006', 'oo22_195a007', 'oo22_195a008', 'oo22_195a009', 'oo22_195a010', 'oo22_195a011', 'oo22_195a012', 'oo22_195a014', 'oo22_195a015', 'oo22_228a003', 'oo22_228a005', 'oo23_181a102', 'oo23_181a105', 'oo23_188a098']
🐳 Finished making label arrays
```

Duration for all labels: ~2min (Apple M3 Pro, 18GB)

## create snippets

```bash
orcai create-snippet-table orcai-v1_recording_table.csv recording_data \
  -p orcai-v1_orcai_parameter.json
```

```console
🐳 Making snippet table
Making snippet tables: 100%|█████████████████| 387/387 [07:30<00:00,  1.17s/recording]
    Created snippet table for 333 recordings.
    Total recording duration: 235:54:10.
    Total number of snippets: 2455200.
    Total number of segments: 4092
    Creating snippet table failed for 54 recordings.
        reason
        missing label files              35
        shorter than segment_duration    19
🐳 Saving snippet table ...
🐳 Snippet table saved to /Volumes/4TB/orcai_project/orcai-v1/recording_data/all_snippets.csv.gz
```

## create tvt snippet tables

```bash
orcai create-tvt-snippet-tables recording_data tvt_data \
  -p orcai-v1_orcai_parameter.json
```

Output:

```console
🐳 Extracting snippets
🐳 Filtering snippet table
    Snippet stats [HMS]:
        data_type      test     train       val     total
        BR         00:15:48  02:24:54  00:18:18  02:59:01
        BUZZ       00:35:08  06:27:13  00:51:15  07:53:37
        HERDING    00:13:24  01:49:56  00:13:48  02:17:09
        PHS        00:02:13  00:41:42  00:04:45  00:48:42
        SS         08:19:46  66:43:43  08:11:39  83:15:09
        TAILSLAP   00:06:43  00:54:14  00:07:19  01:08:17
        WHISTLE    00:02:41  00:23:25  00:02:02  00:28:10
    Percentage of snippets containing no label before selection: 88.52 %
    removing 99.0% of snippets without label
    Percentage of snippets containing no label after selection: 7.16 %
    Number of train, val, test snippets:
        data_type
        test      30255
        train    243227
        val       30059
    Extracting 102400 random train snippets
    Extracting 10240 random val snippets
    Extracting 10240 random test snippets
🐳 Train, val and test snippet tables created and saved to disk
```

## create tvt data

```bash
orcai create-tvt-data tvt_data -p orcai-v1_orcai_parameter.json
```

```console
🐳 Creating train, validation and test data
    Reading in dataframes with snippets and generating loaders
        Data shape:
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1742469420.973148  253800 pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
I0000 00:00:1742469420.973412  253800 pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
            Input spectrogram batch shape: (736, 171, 1)
            Input label batch shape: (46, 7)
        Creating test, validation and training datasets
            Train dataset created. Length 102400.
            Val dataset created. Length 10240.
            Test dataset created. Length 10240.
            Dataset generators created in 00:00:11
🐳 Saving datasets to disk
Saving train dataset: 100%|████████████████████| 102400/102400 [35:50<00:00, 47.61sample/s]
    Size on disk of train_dataset.tfrecord.gz: 11.85 GB
Saving val dataset: 100%|██████████████████████| 10240/10240 [18:48<00:00,  9.07sample/s]
    Size on disk of val_dataset.tfrecord.gz: 1.18 GB
Saving test dataset: 100%|█████████████████████| 10240/10240 [08:20<00:00, 20.46sample/s]
    Size on disk of test_dataset.tfrecord.gz: 1.16 GB
🐳 Train, validation and test datasets created and saved to disk
```

## train model

Run on cluster with GPU.

Setting up environment:

```bash
ssh euler
cd /cluster/home/angstd/orcAI
module load stack/2024-06 gcc/12.2.0 openblas/0.3.24 cuda/12.4.1 python_cuda/3.11.6 py-pip
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_EULER_ROOT
export CUDA_DIR=$CUDA_EULER_ROOT
python -m venv venv
source venv/bin/activate

pip install -U git+https://gitlab.ethz.ch/tb/orcai.git

```

Training:

```bash
#!/usr/bin/bash
#SBATCH --job-name=orcai-v1-dca
#SBATCH --output=orcai-v1-dca/logs/output_%j.log
#SBATCH --error=orcai-v1-dca/logs/error_%j.log
#SBATCH --tmp=15G

#SBATCH --gpus=1
#SBATCH --time=1440
#SBATCH --mem-per-cpu=24g
#SBATCH --gres=gpumem:24g
module load stack/2024-06 gcc/12.2.0 openblas/0.3.24 cuda/12.4.1 python_cuda/3.11.6 py-pip
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_EULER_ROOT
export CUDA_DIR=$CUDA_EULER_ROOT
source /cluster/home/angstd/orcAI/venv/bin/activate

SOURCE_DIR="/cluster/home/angstd/orcAI/tvt_data"

rsync -avz "$SOURCE_DIR/" "$TMPDIR/"

orcai train $TMPDIR /cluster/home/angstd/orcAI -p /cluster/home/angstd/orcAI/orcai-v1-dca_orcai_parameter.json
```

Output:

```console
sending incremental file list
./
dataset_shapes.json
train_dataset.tfrecord.gz
val_dataset.tfrecord.gz

sent 13,038,301,851 bytes  received 76 bytes  199,058,044.69 bytes/sec
total size is 13,035,426,266  speedup is 1.00
🐳 Training model
2025-03-21 15:12:18.434076: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1742566338.648344  458471 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1742566338.705963  458471 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1742566339.190763  458471 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1742566339.190811  458471 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1742566339.190816  458471 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1742566339.190826  458471 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
I0000 00:00:1742566348.340543  458471 gpu_device.cc:2019] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 22807 MB memory:  -> device: 0, name: NVIDIA TITAN RTX, pci bus id: 0000:c1:00.0, compute capability: 7.5

🐳 Loading parameter [0:00:11]
    Output directory: /cluster/home/angstd/orcAI
    Data directory: /scratch/tmp.27096551.angstd
🐳 Loading train, val and test datasets from /scratch/tmp.27096551.angstd [0:00:11, 𝚫 0:00:00]
    Batch size 32
🐳 Building model [0:00:15, 𝚫 0:00:04]
🐳 Building model architecture [0:00:17, 𝚫 0:00:02]
    model name:          orcai-v1-dca
    model architecture:  ResNetLSTM
    model input shape:   (None, 736, 171, 1)
    model output shape:  (None, 46, 7)
    actual input_shape:  (736, 171, 1)
    actual output_shape: (46, 7)
    n_filters:           4
    num_labels:          7
🐳 Compiling model: orcai-v1-dca [0:00:17, 𝚫 0:00:00]
    Learning weights from scratch
    Model size:
        Total parameters: 388935
        Trainable parameters: 387855
        Non-trainable parameters: 1080
        memory usage: 1.68 GB
🐳 Fitting model: orcai-v1-dca [0:00:17, 𝚫 0:00:00]
    Available TensorFlow devices: /CPU:0, /GPU:0
    87%|████████▋ | 26/30 [3:56:10<36:20, 545.04s/epoch, loss=0, masked_binary_accuracy=0, val_loss=0.126, val_masked_binary_accuracy=0.955, learning_rate=3.13e-5]
🐳 Saving Model [3:57:11, 𝚫 3:56:55]
🐳 OrcAI - training model finished. Model saved to orcai-v1-dca.keras [3:57:12, 𝚫 0:00:00]
```

EarlyStopping after 26 epochs.


## test model

```bash
orcai test 
```
