#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p rpi
#SBATCH -A rpi
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --time=25:00:00
#SBATCH --output=%x_out
#SBATCH --error=%x_err

cd /mnt/lustre/rpi/pxu10/manager
source /mnt/lustre/rpi/pxu10/factory/bin/activate
llamafactory-cli train config.yaml