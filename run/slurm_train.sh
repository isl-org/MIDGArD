#!/bin/bash
#SBATCH --job-name=train_sdfusion_mm_partnet
#SBATCH -p g48
#SBATCH --gres=gpu:1
#SBATCH -c 8

source $HOME/miniforge3/bin/activate midgard

./run/shape_generator/train.sh
