#!/bin/bash
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user sussekl@mail.uni-paderborn.de
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=220G
#SBATCH -J "TempME_preprocessing_%a"
#SBATCH -p normal
#SBATCH -A hpc-prf-wiki
#SBATCH --array=0-1
#SBATCH -J "TempME_preprocessing_huge_ram_%A_%a"
#SBATCH -o "Slurm/TempME_preprocessing_huge_ram_%A_%a.out"

module load lang/Python/3.11.5-GCCcore-13.2.0
module load lib/libffi/3.4.4-GCCcore-13.2.0
module load lang/Tkinter/3.11.5-GCCcore-13.2.0
module load system/CUDA/12.4.1
source .tgnn_shap_venv/bin/activate

options=( "UNtrade" "UNvote" )

echo "Running preprocessing for dataset: ${options[$SLURM_ARRAY_TASK_ID]}"

python -m Evaluation.tempme_preprocessing --dataset ${options[$SLURM_ARRAY_TASK_ID]}