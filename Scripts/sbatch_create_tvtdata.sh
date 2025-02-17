#!/bin/bash

module load stack/2024-06 python/3.11.6 py-pip
source myenv/bin/activate

sbatch --mem-per-cpu=16G --time=600 --wrap="MakeData/create_tvtdata.py -c euler  -p /cluster/home/sbonhoef/OrcAI_project/ -m cnn_res_lstm_model"
