#!/bin/bash
#SBATCH --job-name=pnc
#SBATCH --account=simea
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --time=23:50:00
#SBATCH --error=run.err
#SBATCH --output=run_force.out
#SBATCH --gpus-per-node=1
module load TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1
module load Pillow/8.3.2-GCCcore-11.2.0
module load matplotlib/3.4.3-foss-2021b 

python training.py
