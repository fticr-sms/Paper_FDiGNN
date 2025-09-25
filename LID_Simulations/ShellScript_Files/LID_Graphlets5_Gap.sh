#!/bin/bash
#SBATCH --job-name=LID_Graphlet5_Gap
#SBATCH --output=out_LID_graphlets5_Gap.txt
#SBATCH --error=out_error_LID_graphlets5_Gap.txt
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0

ml  OpenMPI/5.0.3-GCC-13.3.0 CUDA/11.8.0 NCCL/2.20.5-GCCcore-13.3.0-CUDA-12.4.0

export PYTHONUSERBASE=/projects/F202406915CPCAA1/pymodules
module load CUDA
module load Anaconda3

CUDA_VISIBLE_DEVICES=0 python LID_Graphlets5_Gap.py 0 &
CUDA_VISIBLE_DEVICES=1 python LID_Graphlets5_Gap.py 1 &
CUDA_VISIBLE_DEVICES=2 python LID_Graphlets5_Gap.py 2 &
CUDA_VISIBLE_DEVICES=3 python LID_Graphlets5_Gap.py 3 &

wait

CUDA_VISIBLE_DEVICES=0 python LID_Graphlets5_Gap.py 4 &
CUDA_VISIBLE_DEVICES=1 python LID_Graphlets5_Gap.py 5 &
CUDA_VISIBLE_DEVICES=2 python LID_Graphlets5_Gap.py 6 &
CUDA_VISIBLE_DEVICES=3 python LID_Graphlets5_Gap.py 7 &

wait

CUDA_VISIBLE_DEVICES=0 python LID_Graphlets5_Gap.py 8 &
CUDA_VISIBLE_DEVICES=1 python LID_Graphlets5_Gap.py 9 &
CUDA_VISIBLE_DEVICES=2 python LID_Graphlets5_Gap.py 10 &
CUDA_VISIBLE_DEVICES=3 python LID_Graphlets5_Gap.py 11 &

wait

CUDA_VISIBLE_DEVICES=0 python LID_Graphlets5_Gap.py 12 &
CUDA_VISIBLE_DEVICES=1 python LID_Graphlets5_Gap.py 13 &
CUDA_VISIBLE_DEVICES=2 python LID_Graphlets5_Gap.py 14 &
CUDA_VISIBLE_DEVICES=3 python LID_Graphlets5_Gap.py 15 &

wait