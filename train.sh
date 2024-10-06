#!/bin/bash
#SBATCH --mem=30G                     # Memory requirement
#SBATCH --gpus=4                      # Request 3 GPUs
#SBATCH -p gpu                        # Use the GPU partition

module load lang/Anaconda3/2020.11
source activate lstm

# Set up proxy environment variables
export http_proxy=http://proxy2.uni-potsdam.de:3128
export https_proxy=http://proxy2.uni-potsdam.de:3128

python train.py

