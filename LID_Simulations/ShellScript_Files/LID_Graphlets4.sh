#!/bin/bash
#SBATCH --job-name=LID_Graphlet4
#SBATCH --output=out_LID_graphlets4_Normal.txt
#SBATCH --error=out_error_LID_graphlets4_Normal.txt
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0

ml  OpenMPI/5.0.3-GCC-13.3.0 CUDA/11.8.0 NCCL/2.20.5-GCCcore-13.3.0-CUDA-12.4.0

export PYTHONUSERBASE=/projects/F202406915CPCAA1/pymodules
module load CUDA
module load Anaconda3

CUDA_VISIBLE_DEVICES=0 python LID_Graphlets4.py 0 &
CUDA_VISIBLE_DEVICES=1 python LID_Graphlets4.py 1 &
CUDA_VISIBLE_DEVICES=2 python LID_Graphlets4.py 2 &
CUDA_VISIBLE_DEVICES=3 python LID_Graphlets4.py 3 &

wait

CUDA_VISIBLE_DEVICES=0 python LID_Graphlets4.py 4 &
CUDA_VISIBLE_DEVICES=1 python LID_Graphlets4.py 5 &
CUDA_VISIBLE_DEVICES=2 python LID_Graphlets4.py 6 &
CUDA_VISIBLE_DEVICES=3 python LID_Graphlets4.py 7 &

wait

CUDA_VISIBLE_DEVICES=0 python LID_Graphlets4.py 8 &
CUDA_VISIBLE_DEVICES=1 python LID_Graphlets4.py 9 &
CUDA_VISIBLE_DEVICES=2 python LID_Graphlets4.py 10 &
CUDA_VISIBLE_DEVICES=3 python LID_Graphlets4.py 11 &

wait

CUDA_VISIBLE_DEVICES=0 python LID_Graphlets4.py 12 &
CUDA_VISIBLE_DEVICES=1 python LID_Graphlets4.py 13 &
CUDA_VISIBLE_DEVICES=2 python LID_Graphlets4.py 14 &
CUDA_VISIBLE_DEVICES=3 python LID_Graphlets4.py 15 &

wait
