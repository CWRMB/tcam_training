#!/bin/bash
#SBATCH --gres=gpu:8
#SBATCH --time=5-06:45:00 # 5days 6hr 45 minutes
#SBATCH --partition=sleipnir5
#SBATCH --output=logs/log-%A.out
#SBATCH --job-name="Tcam-Training"
#SBATCH --mail-type=NONE
#SBATCH --mail-user=tun78940@temple.edu

#Put commands down here
conda activate base

cd /home/tun78940/tcam/tcam_training/traffickcam_model_training
python3 src/train_triplet.py

conda deactivate