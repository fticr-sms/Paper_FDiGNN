#!/bin/bash
#SBATCH --job-name=LID_Graphlet5
#SBATCH --output=out_LID_graphlets5_Normal.txt
#SBATCH --error=out_error_LID_graphlets5_Normal.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0

ml  OpenMPI/5.0.3-GCC-13.3.0 CUDA/11.8.0 NCCL/2.20.5-GCCcore-13.3.0-CUDA-12.4.0

export PYTHONUSERBASE=/projects/F202406915CPCAA1/pymodules
module load CUDA
module load Anaconda3

CUDA_VISIBLE_DEVICES=0 python LID_Graphlets5.py 0 &
CUDA_VISIBLE_DEVICES=1 python LID_Graphlets5.py 1 &
CUDA_VISIBLE_DEVICES=2 python LID_Graphlets5.py 2 &
CUDA_VISIBLE_DEVICES=3 python LID_Graphlets5.py 3 &

wait

CUDA_VISIBLE_DEVICES=0 python LID_Graphlets5.py 4 &
CUDA_VISIBLE_DEVICES=1 python LID_Graphlets5.py 5 &
CUDA_VISIBLE_DEVICES=2 python LID_Graphlets5.py 6 &
CUDA_VISIBLE_DEVICES=3 python LID_Graphlets5.py 7 &

wait

CUDA_VISIBLE_DEVICES=0 python LID_Graphlets5.py 8 &
CUDA_VISIBLE_DEVICES=1 python LID_Graphlets5.py 9 &
CUDA_VISIBLE_DEVICES=2 python LID_Graphlets5.py 10 &
CUDA_VISIBLE_DEVICES=3 python LID_Graphlets5.py 11 &

wait

CUDA_VISIBLE_DEVICES=0 python LID_Graphlets5.py 12 &
CUDA_VISIBLE_DEVICES=1 python LID_Graphlets5.py 13 &
CUDA_VISIBLE_DEVICES=2 python LID_Graphlets5.py 14 &
CUDA_VISIBLE_DEVICES=3 python LID_Graphlets5.py 15 &

wait

python joining_graphlet_dfs.py 16 LID_Graphlets5_Normal_TAG

wait

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

python joining_graphlet_dfs.py 16 LID_Graphlets5_Gap_TAG

wait

CUDA_VISIBLE_DEVICES=0 python LID_Graphlets5_OppGap.py 0 &
CUDA_VISIBLE_DEVICES=1 python LID_Graphlets5_OppGap.py 1 &
CUDA_VISIBLE_DEVICES=2 python LID_Graphlets5_OppGap.py 2 &
CUDA_VISIBLE_DEVICES=3 python LID_Graphlets5_OppGap.py 3 &

wait

CUDA_VISIBLE_DEVICES=0 python LID_Graphlets5_OppGap.py 4 &
CUDA_VISIBLE_DEVICES=1 python LID_Graphlets5_OppGap.py 5 &
CUDA_VISIBLE_DEVICES=2 python LID_Graphlets5_OppGap.py 6 &
CUDA_VISIBLE_DEVICES=3 python LID_Graphlets5_OppGap.py 7 &

wait

CUDA_VISIBLE_DEVICES=0 python LID_Graphlets5_OppGap.py 8 &
CUDA_VISIBLE_DEVICES=1 python LID_Graphlets5_OppGap.py 9 &
CUDA_VISIBLE_DEVICES=2 python LID_Graphlets5_OppGap.py 10 &
CUDA_VISIBLE_DEVICES=3 python LID_Graphlets5_OppGap.py 11 &

wait

CUDA_VISIBLE_DEVICES=0 python LID_Graphlets5_OppGap.py 12 &
CUDA_VISIBLE_DEVICES=1 python LID_Graphlets5_OppGap.py 13 &
CUDA_VISIBLE_DEVICES=2 python LID_Graphlets5_OppGap.py 14 &
CUDA_VISIBLE_DEVICES=3 python LID_Graphlets5_OppGap.py 15 &

wait

python joining_graphlet_dfs.py 16 LID_Graphlets5_OppGap_TAG

wait

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

python joining_graphlet_dfs.py 16 LID_Graphlets4_Normal_TAG

wait