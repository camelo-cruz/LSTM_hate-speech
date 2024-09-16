#!/bin/bash
#SBATCH --job-name=my_gpu_job         # Job name
#SBATCH --mem=20G                     # Memory requirement
#SBATCH --gpus=3                      # Request 3 GPUs
#SBATCH -p gpu                        # Use the GPU partition
#SBATCH --output=job_output.txt       # Output file
#SBATCH --error=job_error.txt         # Error file

module load lang/Anaconda3/2020.11
source activate lstm

# Set up proxy environment variables
export http_proxy=http://proxy2.uni-potsdam.de:3128
export https_proxy=http://proxy2.uni-potsdam.de:3128

python tuning.py

