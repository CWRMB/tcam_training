#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --time=5-06:45:00 # 5days 6hr 45 minutes
#SBATCH --partition=sleipnir3
#SBATCH --output=log-%A.out
#SBATCH --job-name="Tcam-Training"
#SBATCH --mail-type=NONE
#SBATCH --mail-user=tun78940@temple.edu
#SBATCH --cpus-per-task=10

#Put commands down here
#conda activate
cd /home/tun78940/tcam/tcam_training/traffickcam_model_training
singularity exec --nv --bind /shared pytorch_23.03-py3.sif python3 src/resnet_eval.py
#python3 src/resnet_eval.py

