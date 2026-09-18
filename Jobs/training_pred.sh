#!/bin/bash
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user sussekl@mail.uni-paderborn.de
#SBATCH -t 24:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH -J "Training_pred_%A_%a"
#SBATCH -o "Slurm/Training_pred_%A_%a.out"
#SBATCH -p gpu
#SBATCH -A hpc-prf-wiki
#SBATCH --array=0

module load lang/Python/3.9.5-GCCcore-10.3.0
module load lib/libffi/3.3-GCCcore-10.3.0
module load system/CUDA/12.4.1
source .tgnnshap_venv/bin/activate

options=( "Flights" "MOOC" "Reddit" "UNtrade" "UNvote" "USLegis" "Wikipedia" "WikipediaCAWN" )
options=( "RedditCAWN" )

echo "Running prediction for dataset: ${options[$SLURM_ARRAY_TASK_ID]}"

python -u -m Evaluation.Training.prediction -d ${options[$SLURM_ARRAY_TASK_ID]}