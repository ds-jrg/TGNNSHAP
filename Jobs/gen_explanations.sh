#!/bin/bash
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user sussekl@mail.uni-paderborn.de
#SBATCH -t 6:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH -p gpu
#SBATCH -A hpc-prf-wiki
#SBATCH --array=0-6

module load lang/Python/3.11.5-GCCcore-13.2.0
module load lib/libffi/3.4.4-GCCcore-13.2.0
module load lang/Tkinter/3.11.5-GCCcore-13.2.0
module load system/CUDA/12.4.1
source .tgnn_shap_venv/bin/activate

options=( "Flights" "MOOC" "Reddit" "UNtrade" "UNvote" "USLegis" "Wikipedia" )

echo "Running for dataset: ${options[$SLURM_ARRAY_TASK_ID]}"

python -m Evaluation.misc_tests.high_quality --dataset ${options[$SLURM_ARRAY_TASK_ID]} 
