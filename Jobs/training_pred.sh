#!/bin/bash
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user sussekl@mail.uni-paderborn.de
#SBATCH -t 5:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH -J "Shapley_Experiment_%a"
#SBATCH -p gpu
#SBATCH -A hpc-prf-wiki
#SBATCH --array=0-7

module load lang/Python/3.9.5-GCCcore-10.3.0
module load lib/libffi/3.4.5-GCCcore-13.3.0
module load system/CUDA/12.4.1
source venv/bin/activate

options=( "Flights" "MOOC" "Reddit" "UNtrade" "UNvote" "USLegis" "Wikipedia" )

python -u -m Evaluation.Training.prediction -d ${options[$SLURM_ARRAY_TASK_ID]}