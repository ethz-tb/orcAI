# orcai-v1 pipeline

## install in a virtual environment

```bash
pyenv virtualenv 3.11 orcai
pyenv activate orcai
python -m pip install --upgrade pip
pip install -U git+https://github.com/ethz-tb/orcAI.git
```

```bash
orcai --version
```

```console
orcai, version 0.19.0
```


## Initialize project

```bash
orcai init orcai orcai
cd orcai
```

```console
🐳 Initializing project
orcAI 0.19.0 [started @ 2025-04-24 09:32:01]
🐳 Creating project directory: /Volumes/4TB/orcai_project/orcai [0:00:28]
    Creating orcai_orcai_parameter.json
    Creating orcai_hps_parameter.json
    Creating orcai_call_duration_limits.json
    Generating random seed
🐳 Project initialized. [0:00:29, 𝚫 0:00:01]
```

## create/update recording table

create recording table from directory with recordings.
update original_recording_table (containing possibilities of calls entered by Chérine).
Calls to label are in orcai-v1_orcai_parameter.json.
Files to exclude are in orcai-v1_files_exclude.json.

```bash
orcai create-recording-table ../orca_recordings/Acoustics \
-o ../orca_recordings/recording_table.csv \
-ut ../orca_recordings/original_recording_table.csv \
-p orcai_parameter.json \
-ep files_exclude.json \
-up
```

```console
🐳 Creating recording table
orcAI 0.19.0 [started @ 2025-04-24 10:03:23]
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
    ‼️ 4 annotations with missing recordings: {'oo21_202b007', 'oo21_184a007', 'oo21_189a011', 'oo21_184a006'}. These will be ignored.
🐳 Saving recording table to /Volumes/4TB/orcai_project/orca_recordings/recording_table.csv [0:00:03, 𝚫 0:00:00]
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
orcai create-spectrograms ../orca_recordings/recording_table.csv \
../orca_recordings/recording_data \
-p orcai_parameter.json

```

```console
🐳 Creating spectrograms
orcAI 0.19.0 [started @ 2025-04-24 10:11:18]
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
    Skipping 0 recordings because they already have spectrograms.
🐳 Creating 342 spectrograms [0:00:05, 𝚫 0:00:01]
Making spectrograms: 100%|████████████████████| 342/342 [59:39<00:00, 10.47s/it]
🐳 Spectrograms created. [0:59:45, 𝚫 0:59:40]
```

## create label arrays

Create all label arrasy for recordings in recording_table and save to
recording_data.
Use orcai_parameter.json parameters.
Unify call labels according call_equivalences.json.

```bash
orcai create-label-arrays ../orca_recordings/recording_table.csv \
../orca_recordings/recording_data \
-p orcai_parameter.json \
-ce call_equivalences.json
```

```console
🐳 Creating label arrays
orcAI 0.19.0 [started @ 2025-04-24 11:13:15]
🐳 Reading recordings table [0:00:02]
    Skipping 574 because of missing annotation files.
🐳 Making label arrays [0:00:02, 𝚫 0:00:00]
Making label arrays: 100%|█████████████| 387/387 [00:58<00:00,  6.63recording/s]
    ‼️ No valid labels present in ['2015-07-29c', '2015-12-07l', '2015-13-07b',
    '2015-13-07c', '2015-17-07c', '2015-17-07h', '2015-18-07a', '2015-18-07f',
    '2015-21-07g', '2015-25-07f', '2015-25-07i', '2015-25-07j', '2015_07_14b',
    '2016-05-07C', '2016-12-07C', '2016-13-07K', '2016-16-07I',
    '2016-20-071008', '2016-24-07T323', '2016-24-07T328', '2016-25-07T338',
    '2016-27-07T352', '2016-27-07T363', 'oo09_200a043', 'oo09_209a012',
    'oo14_048a012', 'oo21_175a004', 'oo21_182a004', 'oo21_184a020',
    'oo22_195a004', 'oo22_195a005', 'oo22_195a006', 'oo22_195a007',
    'oo22_195a008', 'oo22_195a009', 'oo22_195a010', 'oo22_195a011',
    'oo22_195a012', 'oo22_195a014', 'oo22_195a015', 'oo22_228a003',
    'oo22_228a005', 'oo23_181a102', 'oo23_181a105', 'oo23_188a098']
🐳 Finished making label arrays [0:01:01, 𝚫 0:00:58]
```

## create snippets

```bash
orcai create-snippet-table ../orca_recordings/recording_table.csv \
../orca_recordings/recording_data \
-p orcai_parameter.json
```

```console
🐳 Creating snippet table
orcAI 0.19.0 [started @ 2025-04-28 13:34:44]
🐳 Reading recording table [0:00:03]
    ‼️ Missing recording data directories for 45 recordings. Skipping these recordings.
    ‼️ Did you create the spectrograms & Labels?
🐳 Making snippet tables [0:00:03, 𝚫 0:00:00]
Making snippet tables: 100%|███████████| 342/342 [21:55<00:00,  3.85s/recording]
    Created snippet table for 333 recordings.
    Total recording duration: 235:54:10.
    Total number of snippets: 2434541.
    Total number of segments: 4092
    Creating snippet table failed for 9 recordings.
        reason
        shorter than segment_duration    9
🐳 Saving snippet table... [0:21:59, 𝚫 0:21:56]
🐳 Snippet table saved to ../orca_recordings/tvt_data/all_snippets.csv.gz [0:22:10, 𝚫 0:00:11]
```

## create tvt snippet tables

```bash
orcai create-tvt-snippet-tables ../orca_recordings/tvt_data \
-p orcai_parameter.json
```

```console
🐳 Creating train, validation and test snippet tables
orcAI 0.19.0 [started @ 2025-04-28 13:58:31]
🐳 Reading snippet table [0:00:04]
    Snippet stats [HMS]:
        data_type     train       val      test     total
        BR         02:21:17  00:17:41  00:16:09  02:55:08
        BUZZ       06:18:07  00:48:16  00:35:34  07:41:57
        HERDING    01:39:27  00:13:32  00:12:29  02:05:28
        PHS        00:43:55  00:04:46  00:02:38  00:51:21
        SS         66:12:58  08:14:02  08:19:46  82:46:47
        TAILSLAP   00:51:30  00:07:25  00:06:54  01:05:51
        WHISTLE    00:20:01  00:02:23  00:02:44  00:25:09
🐳 Filtering snippet table [0:00:05, 𝚫 0:00:01]
    Percentage of snippets containing no label before selection: 88.52 %
    removing 99.0% of snippets without label
    Percentage of snippets containing no label after selection: 7.16 %
    Number of train, val, test snippets:
        data_type
        test      30154
        train    241024
        val       29940
    Extracting 3750 batches of 64 random train snippets (240000 snippets)
    Extracting 375 batches of 64 random val snippets (24000 snippets)
    Extracting 375 batches of 64 random test snippets (24000 snippets)
    Snippet stats for train, val and test datasets [HMS]:
        data_type     train       val      test     total
        BR         02:20:39  00:14:14  00:12:57  02:47:51
        BUZZ       06:16:55  00:38:17  00:29:04  07:24:17
        HERDING    01:39:11  00:10:51  00:10:07  02:00:09
        PHS        00:43:46  00:03:41  00:02:06  00:49:34
        SS         65:55:49  06:37:10  06:37:12  79:10:12
        TAILSLAP   00:51:26  00:05:45  00:05:33  01:02:44
        WHISTLE    00:19:57  00:01:50  00:02:08  00:23:56
🐳 Train, val and test snippet tables created and saved to disk [0:00:06, 𝚫 0:00:01]
```

## create tvt data

```bash
orcai orcai create-tvt-data ../orca_recordings/tvt_data -p orcai_parameter.json
```

```console
🐳 Creating train, validation and test datasets
orcAI 0.21.0 [started @ 2025-04-29 09:52:57]
🐳 Reading in snippet tables and generating loaders [0:00:03]
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1745913367.073332  357666 pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
I0000 00:00:1745913367.073522  357666 pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
    Data shape:
        Input spectrogram batch shape: (736, 171, 1)
        Input label batch shape: (46, 7)
🐳 Creating test, validation and training datasets [0:03:09, 𝚫 0:03:06]
    Train dataset created. Length 240000.
    Val dataset created. Length 24000.
    Test dataset created. Length 24000.
🐳 Saving datasets to disk [0:03:09, 𝚫 0:00:00]
    Size on disk of train_dataset: 27.79 GB
    Size on disk of val_dataset: 2.72 GB
    Size on disk of test_dataset: 2.71 GB
🐳 Train, validation and test datasets created and saved to disk [1:19:18, 𝚫 1:16:08]
```

## running hyperparameter search

Run on ETHZ Euler cluster with GPU.

Setting up environment:

```bash
ssh euler
cd /cluster/home/angstd/orcAI
module load stack/2024-06 gcc/12.2.0 openblas/0.3.24 cuda/12.4.1 python_cuda/3.11.6 py-pip
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_EULER_ROOT
export CUDA_DIR=$CUDA_EULER_ROOT
python -m venv venv
source venv/bin/activate

pip install -U git+https://github.com/ethz-tb/orcAI.git
orcai --version
```

```bash
#!/usr/bin/bash
#SBATCH --job-name=orcai-hpsearch
#SBATCH --output=/cluster/home/angstd/orcAI/20250428_orcai/logs/hpsearch_%j.log
#SBATCH --tmp=80G

#SBATCH --gpus=4
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=24g
#SBATCH --gres=gpumem:24g
module load stack/2024-06 gcc/12.2.0 openblas/0.3.24 cuda/12.4.1 python_cuda/3.11.6 py-pip
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_EULER_ROOT
export CUDA_DIR=$CUDA_EULER_ROOT
source /cluster/home/angstd/orcAI/venv/bin/activate

SOURCE_FILE="/nfs/nas22/fs2201/usys_ibz_tb_data/TB-data/Daniel_Angst/orcAI/orcai_tvt_data_3750.zip"
filename=$(basename -- "$SOURCE_FILE")

rsync_cmd="rsync -avz $SOURCE_FILE $TMPDIR"
echo -e "\nrunning $rsync_cmd\n"
eval $rsync_cmd

unzip_cmd="unzip $TMPDIR/$filename -d $TMPDIR"
echo -e "\nrunning $unzip_cmd\n"
eval $unzip_cmd

data_dir="$TMPDIR"
output_dir="/cluster/home/angstd/orcAI/20250428_orcai"
parameter_file="$output_dir/orcai_parameter.json"
hps_parameter_file="$output_dir/hps_parameter.json"

hp_search_cmd="orcai hpsearch $data_dir $output_dir -p $parameter_file -hp $hps_parameter_file -pl"
echo -e "\nrunning $hp_search_cmd\n"
orcai --version
eval $hp_search_cmd
```

```console
Many modules are hidden in this stack. Use "module --show_hidden spider SOFTWARE" if you are not able to find the required software

running rsync -avz /nfs/nas22/fs2201/usys_ibz_tb_data/TB-data/Daniel_Angst/orcAI/orcai_tvt_data_3750.zip /scratch/tmp.30631690.angstd

sending incremental file list
orcai_tvt_data_3750.zip

sent 33,226,309,469 bytes  received 35 bytes  64,205,428.99 bytes/sec
total size is 33,218,790,656  speedup is 1.00

running unzip /scratch/tmp.30631690.angstd/orcai_tvt_data_3750.zip -d /scratch/tmp.30631690.angstd

Archive:  /scratch/tmp.30631690.angstd/orcai_tvt_data_3750.zip

running orcai hpsearch /scratch/tmp.30631690.angstd /cluster/home/angstd/orcAI/20250428_orcai -p /cluster/home/angstd/orcAI/20250428_orcai/orcai_parameter_HPS.json -hp /cluster/home/angstd/orcAI/20250428_orcai/hps_parameter.json -pl

orcai, version 0.22.1
🐳 Hyperparameter search
orcAI 0.22.1 [started @ 2025-05-02 10:53:12]
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1746175993.820539  322561 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1746175993.827003  322561 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1746175993.848934  322561 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746175993.848983  322561 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746175993.848988  322561 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746175993.848992  322561 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
🐳 Loading Hyperparameter search parameter [0:00:28]
🐳 Loading training and validation datasets from /scratch/tmp.30631690.angstd [0:00:28, 𝚫 0:00:00]
I0000 00:00:1746176023.107064  322561 gpu_device.cc:2019] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 22807 MB memory:  -> device: 0, name: Quadro RTX 6000, pci bus id: 0000:01:00.0, compute capability: 7.5
I0000 00:00:1746176023.112165  322561 gpu_device.cc:2019] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 22807 MB memory:  -> device: 1, name: Quadro RTX 6000, pci bus id: 0000:21:00.0, compute capability: 7.5
I0000 00:00:1746176023.113749  322561 gpu_device.cc:2019] Created device /job:localhost/replica:0/task:0/device:GPU:2 with 22807 MB memory:  -> device: 2, name: Quadro RTX 6000, pci bus id: 0000:22:00.0, compute capability: 7.5
I0000 00:00:1746176023.115533  322561 gpu_device.cc:2019] Created device /job:localhost/replica:0/task:0/device:GPU:3 with 22807 MB memory:  -> device: 3, name: Quadro RTX 6000, pci bus id: 0000:81:00.0, compute capability: 7.5
🐳 Searching hyperparameters [0:00:34, 𝚫 0:00:07]
    Parallel - running on 4 GPU
Reloading Tuner from /cluster/home/angstd/orcAI/20250428_orcai/hps_logs/orcai-v1-3750-LSTM_HPS/tuner0.json
    Saving best model to hps/orcai-v1-3750-LSTM_HPS.keras
I0000 00:00:1746176045.407298  322656 cuda_dnn.cc:529] Loaded cuDNN version 90300
/cluster/home/angstd/orcAI/venv/lib/python3.11/site-packages/keras/src/saving/saving_lib.py:757: UserWarning: Skipping variable loading for optimizer 'adam', because it has 2 variables whereas the saved optimizer has 152 variables. 
  saveable.load_own_variables(weights_store.get(inner_path))
🐳 Best Hyperparameters [2 days, 10:29:01, 𝚫 2 days, 10:28:27]
    {
        "filters": "set3",
        "kernel_size": 3,
        "dropout_rate": 0.5,
        "batch_size": 64,
        "lstm_units": 128,
        "tuner/epochs": 10,
        "tuner/initial_epoch": 4,
        "tuner/bracket": 1,
        "tuner/round": 1,
        "tuner/trial_id": "0018"
    }
🐳 Hyperparameter search completed [2 days, 10:29:01, 𝚫 0:00:00]
```

## train model

Run on ETHZ Euler cluster with GPU.


Training:

```bash
#!/usr/bin/bash
#SBATCH --job-name=orcai-v1
#SBATCH --output=/cluster/home/angstd/orcAI/orcai-v1/logs/training_output_%j.log
#SBATCH --tmp=80G

#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=24g
#SBATCH --gres=gpumem:24g
module load stack/2024-06 gcc/12.2.0 openblas/0.3.24 cuda/12.4.1 python_cuda/3.11.6 py-pip
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_EULER_ROOT
export CUDA_DIR=$CUDA_EULER_ROOT
source /cluster/home/angstd/orcAI/venv/bin/activate

SOURCE_FILE="/nfs/nas22/fs2201/usys_ibz_tb_data/TB-data/Daniel_Angst/orcAI/orcai_tvt_data_3750.zip"
filename=$(basename -- "$SOURCE_FILE")

rsync_cmd="rsync -avz $SOURCE_FILE $TMPDIR"
echo -e "\nrunning $rsync_cmd\n"
eval $rsync_cmd

unzip_cmd="unzip $TMPDIR/$filename -d $TMPDIR"
echo -e "\nrunning $unzip_cmd\n"
eval $unzip_cmd

data_dir="$TMPDIR"
output_dir="/cluster/home/angstd/orcAI"
parameter_file="$output_dir/orcai_parameter_v1.json"

orcai_train_cmd="orcai train $data_dir $output_dir -p $parameter_file"
echo -e "\nrunning $orcai_train_cmd\n"
orcai --version
eval $orcai_train_cmd
```

```console
🐳 Training model
orcAI 0.23.0 [started @ 2025-05-05 08:32:27]
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1746426748.478035  628398 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1746426748.484835  628398 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1746426748.506009  628398 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746426748.506051  628398 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746426748.506055  628398 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746426748.506059  628398 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    Platform: Linux-5.15.0-131-generic-x86_64-with-glibc2.35
    Python version: 3.11.6 (main, Jun 20 2024, 16:12:48) [GCC 12.2.0]
    Tensorflow version: 2.19.0
    Keras version: 3.9.2
    CUDA version: 12.5.1
    cuDNN version: 9
    Available TensorFlow devices: /GPU:0: Quadro RTX 6000
🐳 Loading parameter [0:00:27]
    Output directory: /cluster/home/angstd/orcAI
    Data directory: /scratch/tmp.30907263.angstd
🐳 Loading training and validation datasets from /scratch/tmp.30907263.angstd [0:00:27, 𝚫 0:00:00]
    Loading dataset shapes from JSON file
I0000 00:00:1746426774.337002  628398 gpu_device.cc:2019] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 22807 MB memory:  -> device: 0, name: Quadro RTX 6000, pci bus id: 0000:81:00.0, compute capability: 7.5
    Batch size 64
🐳 Building model [0:00:31, 𝚫 0:00:05]
🐳 Building model architecture [0:00:33, 𝚫 0:00:02]
    model name:          orcai-v1
    model architecture:  ResNetLSTM
    model input shape:   (None, 736, 171, 1)
    model output shape:  (None, 46, 7)
    actual input_shape:  (736, 171, 1)
    actual output_shape: (46, 7)
    n_filters:           4
    num_labels:          7
🐳 Compiling model: orcai-v1 [0:00:33, 𝚫 0:00:00]
    Model size:
        Total parameters: 996039
        Trainable parameters: 994959
        Non-trainable parameters: 1080
        memory usage: 1.67 GB
🐳 Fitting model: orcai-v1 [0:00:33, 𝚫 0:00:00]
    Monitoring val_MBA

0epoch [00:00, ?epoch/s]
  0%|          | 0/30 [00:00<?, ?epoch/s]I0000 00:00:1746426793.368161  628472 cuda_dnn.cc:529] Loaded cuDNN version 90300

  3%|▎         | 1/30 [32:50<15:52:29, 1970.66s/epoch, MBA=0.922, loss=0.644, val_MBA=0.95, val_loss=0.249, learning_rate=0.0001]
  7%|▋         | 2/30 [1:06:00<15:24:58, 1982.10s/epoch, MBA=0.951, loss=0.202, val_MBA=0.954, val_loss=0.162, learning_rate=0.0001]
 10%|█         | 3/30 [1:39:52<15:02:06, 2004.68s/epoch, MBA=0.956, loss=0.143, val_MBA=0.955, val_loss=0.134, learning_rate=0.0001]
 13%|█▎        | 4/30 [2:12:45<14:23:19, 1992.29s/epoch, MBA=0.959, loss=0.121, val_MBA=0.958, val_loss=0.121, learning_rate=0.0001]
 17%|█▋        | 5/30 [2:46:06<13:51:23, 1995.34s/epoch, MBA=0.962, loss=0.109, val_MBA=0.958, val_loss=0.119, learning_rate=5e-5]  
 20%|██        | 6/30 [3:19:29<13:19:11, 1997.97s/epoch, MBA=0.963, loss=0.105, val_MBA=0.959, val_loss=0.119, learning_rate=5e-5]
 23%|██▎       | 7/30 [3:52:53<12:46:36, 1999.84s/epoch, MBA=0.964, loss=0.102, val_MBA=0.958, val_loss=0.12, learning_rate=5e-5] 
 27%|██▋       | 8/30 [4:25:23<12:07:29, 1984.08s/epoch, MBA=0.966, loss=0.0962, val_MBA=0.959, val_loss=0.118, learning_rate=2.5e-5]
 30%|███       | 9/30 [4:57:54<11:30:50, 1973.83s/epoch, MBA=0.966, loss=0.0941, val_MBA=0.959, val_loss=0.119, learning_rate=2.5e-5]
 33%|███▎      | 10/30 [5:31:13<11:00:30, 1981.51s/epoch, MBA=0.967, loss=0.0923, val_MBA=0.959, val_loss=0.119, learning_rate=2.5e-5]
 37%|███▋      | 11/30 [6:04:37<10:29:40, 1988.43s/epoch, MBA=0.968, loss=0.0894, val_MBA=0.959, val_loss=0.121, learning_rate=1.25e-5]
 40%|████      | 12/30 [6:37:58<9:57:40, 1992.25s/epoch, MBA=0.968, loss=0.0882, val_MBA=0.959, val_loss=0.121, learning_rate=1.25e-5] 
 43%|████▎     | 13/30 [7:10:28<9:20:50, 1979.44s/epoch, MBA=0.969, loss=0.0873, val_MBA=0.959, val_loss=0.123, learning_rate=1.25e-5]
 47%|████▋     | 14/30 [7:43:49<8:49:34, 1985.88s/epoch, MBA=0.969, loss=0.0858, val_MBA=0.959, val_loss=0.123, learning_rate=6.25e-6]
 50%|█████     | 15/30 [8:17:11<8:17:43, 1990.89s/epoch, MBA=0.97, loss=0.0851, val_MBA=0.959, val_loss=0.124, learning_rate=6.25e-6] 
 53%|█████▎    | 16/30 [8:49:41<7:41:40, 1978.59s/epoch, MBA=0.97, loss=0.0848, val_MBA=0.959, val_loss=0.123, learning_rate=6.25e-6]
 57%|█████▋    | 17/30 [9:22:11<7:06:48, 1969.90s/epoch, MBA=0.97, loss=0.0838, val_MBA=0.959, val_loss=0.125, learning_rate=3.12e-6]
 60%|██████    | 18/30 [9:55:33<6:35:55, 1979.63s/epoch, MBA=0.97, loss=0.0836, val_MBA=0.959, val_loss=0.125, learning_rate=3.12e-6]
 63%|██████▎   | 19/30 [10:28:58<6:04:19, 1987.20s/epoch, MBA=0.97, loss=0.0833, val_MBA=0.959, val_loss=0.125, learning_rate=3.12e-6]
 67%|██████▋   | 20/30 [11:02:20<5:31:56, 1991.60s/epoch, MBA=0.97, loss=0.0829, val_MBA=0.959, val_loss=0.125, learning_rate=1.56e-6]
 70%|███████   | 21/30 [11:34:51<4:56:55, 1979.45s/epoch, MBA=0.97, loss=0.0827, val_MBA=0.959, val_loss=0.125, learning_rate=1.56e-6]
 73%|███████▎  | 22/30 [12:07:51<4:23:55, 1979.47s/epoch, MBA=0.97, loss=0.0826, val_MBA=0.959, val_loss=0.126, learning_rate=1.56e-6]
 77%|███████▋  | 23/30 [12:40:43<3:50:40, 1977.21s/epoch, MBA=0.971, loss=0.0823, val_MBA=0.959, val_loss=0.126, learning_rate=7.81e-7]
 80%|████████  | 24/30 [13:13:10<3:16:50, 1968.34s/epoch, MBA=0.971, loss=0.0824, val_MBA=0.959, val_loss=0.126, learning_rate=7.81e-7]
 80%|████████  | 24/30 [13:13:10<3:18:17, 1982.95s/epoch, MBA=0.971, loss=0.0824, val_MBA=0.959, val_loss=0.126, learning_rate=7.81e-7]
🐳 Saving Model [13:14:38, 𝚫 13:14:05]
🐳 Training model finished. Model saved to orcai-v1.keras [13:14:39, 𝚫 0:00:00]
```

## test model

```bash
orcai test 
```
