#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p qTRDGPU
#SBATCH -t 1440
#SBATCH -J Tishby
#SBATCH -e logs/err%A-%a.err
#SBATCH -o logs/out%A-%a.out
#SBATCH -A PSYC0002
#SBATCH --oversubscribe
#SBATCH --gpus=1
export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/bin:/usr/local/cuda/lib64/
eval "$(conda shell.bash hook)"
conda activate tf
cd /data/mialab/users/bbaker/projects/InfoBound
CUDA_AVAILABLE_DEVICES=0 python main.py -learning_rate 0.0001 -num_epochs 1000 -num_repeat 1 -d_name MNIST -cov_net 1
