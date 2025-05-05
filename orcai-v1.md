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

model_dir="$output_dir/orcai-v1"

orcai_test_cmd="orcai test $model_dir $data_dir"
echo -e "\nrunning $orcai_test_cmd\n"
eval $orcai_test_cmd
```

## test model

```bash
orcai test 
```
