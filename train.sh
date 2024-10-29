#!/bin/bash -l

# Set SCC project
#$ -P dnn-motion

# Request 16 CPUs
#$ -pe omp 16

# Request 1 GPU 
#$ -l gpus=1

# Specify the minimum GPU compute capability. 
#$ -l gpu_c=8.0


module load miniconda

conda activate anerf
python run_nerf.py --basedir logs --expname rats_model