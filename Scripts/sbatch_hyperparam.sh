#!/usr/bin/bash
#SBATCH --job-name=hyp  # Job name
#SBATCH --output=Results/cnn_res_lstm_model/hyperpar/output_%j.out      # Log file (%j is the job ID)
#SBATCH --error=Results/cnn_res_lstm_model/hyperpar/error_%j.out       # Error file (%j is the job ID)
#SBATCH --gpus=8
#SBATCH --time=1400
#SBATCH --mem-per-cpu=24g
#SBATCH --tmp=100G
#SBATCH --gres=gpumem:24g


module load stack/2024-06 python_cuda/3.9.18 py-pip
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/cluster/apps/gcc-8.2.0/cuda-11.8.0-pbjfn3bg7evmbqooqn5j625abh3dxc3e
source myenv/bin/activate

SOURCE_DIR="/cluster/scratch/sbonhoef/tvtdata"

# Copy directory using rsync
rsync -avz "$SOURCE_DIR/" "$TMPDIR/"

# Or use cp command
# cp -r "$SOURCE_DIR/" "$TMPDIR/"

#echo "Copied $SOURCE_DIR to $TMPDIR"

Train_and_Test/hyperparameter_search.py -c euler -m cnn_res_lstm_model -d "$TMPDIR/" -p /cluster/home/sbonhoef/OrcAI_project/ 

