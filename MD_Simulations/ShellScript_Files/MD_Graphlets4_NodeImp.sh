#!/bin/bash
#SBATCH --job-name=MD_Graphlet4_TAG_NodeImp
#SBATCH --output=out_MD_graphlets4_Normal_TAG_NodeImp.txt
#SBATCH --error=out_error_MD_graphlets4_Normal_TAG_NodeImp.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0

ml  OpenMPI/5.0.3-GCC-13.3.0 CUDA/11.8.0 NCCL/2.20.5-GCCcore-13.3.0-CUDA-12.4.0

export PYTHONUSERBASE=/projects/F202406915CPCAA1/pymodules
module load CUDA
module load Anaconda3

CUDA_VISIBLE_DEVICES=0 python MD_Graphlets4_NodeImp.py 0 &
CUDA_VISIBLE_DEVICES=1 python MD_Graphlets4_NodeImp.py 1 &
CUDA_VISIBLE_DEVICES=2 python MD_Graphlets4_NodeImp.py 2 &
CUDA_VISIBLE_DEVICES=3 python MD_Graphlets4_NodeImp.py 3 &

wait

CUDA_VISIBLE_DEVICES=0 python MD_Graphlets4_NodeImp.py 4 &
CUDA_VISIBLE_DEVICES=1 python MD_Graphlets4_NodeImp.py 5 &
CUDA_VISIBLE_DEVICES=2 python MD_Graphlets4_NodeImp.py 6 &
CUDA_VISIBLE_DEVICES=3 python MD_Graphlets4_NodeImp.py 7 &

wait

CUDA_VISIBLE_DEVICES=0 python MD_Graphlets4_NodeImp.py 8 &
CUDA_VISIBLE_DEVICES=1 python MD_Graphlets4_NodeImp.py 9 &
CUDA_VISIBLE_DEVICES=2 python MD_Graphlets4_NodeImp.py 10 &
CUDA_VISIBLE_DEVICES=3 python MD_Graphlets4_NodeImp.py 11 &

wait

CUDA_VISIBLE_DEVICES=0 python MD_Graphlets4_NodeImp.py 12 &
CUDA_VISIBLE_DEVICES=1 python MD_Graphlets4_NodeImp.py 13 &
CUDA_VISIBLE_DEVICES=2 python MD_Graphlets4_NodeImp.py 14 &
CUDA_VISIBLE_DEVICES=3 python MD_Graphlets4_NodeImp.py 15 &

wait

python joining_graphlet_dfs.py 16 MD_Graphlets4_Normal_TAG