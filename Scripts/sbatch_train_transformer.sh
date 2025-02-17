#!/usr/bin/bash
#SBATCH --job-name=trafo
#SBATCH --output=Results/cnn_res_transformer_model/logs/output_%j.log
#SBATCH --error=Results/cnn_res_transformer_model/logs/error_%j.log

#SBATCH --gpus=1
#SBATCH --time=1440
#SBATCH --mem-per-cpu=24g
#SBATCH --tmp=100G
#SBATCH --gres=gpumem:24g


module load stack/2024-06 python_cuda/3.9.18 py-pip
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/cluster/apps/gcc-8.2.0/cuda-11.8.0-pbjfn3bg7evmbqooqn5j625abh3dxc3e
source myenv/bin/activate

SOURCE_DIR="/cluster/scratch/sbonhoef/tvtdata"

# Copy directory using rsync
rsync -avz "$SOURCE_DIR/" "$TMPDIR/"

Train_and_Test/train_model.py -c euler -m cnn_res_transformer_model -d "$TMPDIR/" -p /cluster/home/sbonhoef/OrcAI_project/ 




