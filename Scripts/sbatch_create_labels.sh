#!/bin/bash

module load stack/2024-06 python/3.11.6 py-pip
source myenv/bin/activate

sbatch --mem-per-cpu=16G --time=60 --wrap="MakeData/create_labels.py -c euler  -p /cluster/home/sbonhoef/OrcAI_project/"
