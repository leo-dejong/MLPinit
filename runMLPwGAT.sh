#!/bin/bash
#SBATCH --job-name=poopy       
#SBATCH --nodes=1                  
#SBATCH --ntasks-per-node=1          
#SBATCH --partition=gpu              
#SBATCH --gpus=2              
#SBATCH --cpus-per-gpu=2             
#SBATCH --time=1:00:00              
#SBATCH --output=output.log
#SBATCH --mem=64G  

module load miniconda
conda activate mlpinit
module load cuDNN/8.8.0.121-CUDA-12.0.0

python MLPwGAT.py



