#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --time=5-06:45:00 # 5days 6hr 45 minutes
#SBATCH --partition=sleipnir3
#SBATCH --output=log-%A.out
#SBATCH --job-name="Tcam-Training"
#SBATCH --mail-type=NONE
#SBATCH --mail-user=tun78940@temple.edu
 
#Put commands down here
# . miniconda3/etc/profile.d/conda.sh
#conda init bash
#conda activate
cd /home/tun78940/tcam/tcam_training/traffickcam_model_training
python3 src/train.py

