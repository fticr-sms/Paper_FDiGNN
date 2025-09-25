#!/bin/bash
#SBATCH --job-name=LGD_Graphlets5_Normal
#SBATCH --output=out_LGD_graphlets5.txt
#SBATCH --error=out_error_LGD_graphlets5.txt
#SBATCH --time=010:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0

export PYTHONUSERBASE=/projects/F202406915CPCAA1/pymodules
module load CUDA
module load Anaconda3

python LGD_5Graphlets_Normal.py 0 &
python LGD_5Graphlets_Normal.py 1 &
python LGD_5Graphlets_Normal.py 2 &
python LGD_5Graphlets_Normal.py 3 &

wait

python LGD_5Graphlets_Normal.py 4 &
python LGD_5Graphlets_Normal.py 5 &
python LGD_5Graphlets_Normal.py 6 &
python LGD_5Graphlets_Normal.py 7 &

wait

python LGD_5Graphlets_Normal.py 8 &
python LGD_5Graphlets_Normal.py 9 &
python LGD_5Graphlets_Normal.py 10 &
python LGD_5Graphlets_Normal.py 11 &

wait

python LGD_5Graphlets_Normal.py 12 &
python LGD_5Graphlets_Normal.py 13 &
python LGD_5Graphlets_Normal.py 14 &
python LGD_5Graphlets_Normal.py 15 &

wait

python LGD_5Graphlets_Normal.py 16 &
python LGD_5Graphlets_Normal.py 17 &
python LGD_5Graphlets_Normal.py 18 &
python LGD_5Graphlets_Normal.py 19 &

wait

python LGD_5Graphlets_Gap.py 0 &
python LGD_5Graphlets_Gap.py 1 &
python LGD_5Graphlets_Gap.py 2 &
python LGD_5Graphlets_Gap.py 3 &

wait

python LGD_5Graphlets_Gap.py 4 &
python LGD_5Graphlets_Gap.py 5 &
python LGD_5Graphlets_Gap.py 6 &
python LGD_5Graphlets_Gap.py 7 &

wait

python LGD_5Graphlets_Gap.py 8 &
python LGD_5Graphlets_Gap.py 9 &
python LGD_5Graphlets_Gap.py 10 &
python LGD_5Graphlets_Gap.py 11 &

wait

python LGD_5Graphlets_Gap.py 12 &
python LGD_5Graphlets_Gap.py 13 &
python LGD_5Graphlets_Gap.py 14 &
python LGD_5Graphlets_Gap.py 15 &

wait

python LGD_5Graphlets_Gap.py 16 &
python LGD_5Graphlets_Gap.py 17 &
python LGD_5Graphlets_Gap.py 18 &
python LGD_5Graphlets_Gap.py 19 &

wait

python LGD_5Graphlets_OppGap.py 0 &
python LGD_5Graphlets_OppGap.py 1 &
python LGD_5Graphlets_OppGap.py 2 &
python LGD_5Graphlets_OppGap.py 3 &

wait

python LGD_5Graphlets_OppGap.py 4 &
python LGD_5Graphlets_OppGap.py 5 &
python LGD_5Graphlets_OppGap.py 6 &
python LGD_5Graphlets_OppGap.py 7 &

wait

python LGD_5Graphlets_OppGap.py 8 &
python LGD_5Graphlets_OppGap.py 9 &
python LGD_5Graphlets_OppGap.py 10 &
python LGD_5Graphlets_OppGap.py 11 &

wait

python LGD_5Graphlets_OppGap.py 12 &
python LGD_5Graphlets_OppGap.py 13 &
python LGD_5Graphlets_OppGap.py 14 &
python LGD_5Graphlets_OppGap.py 15 &

wait

python LGD_5Graphlets_OppGap.py 16 &
python LGD_5Graphlets_OppGap.py 17 &
python LGD_5Graphlets_OppGap.py 18 &
python LGD_5Graphlets_OppGap.py 19 &

wait