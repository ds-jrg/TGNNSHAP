#!/bin/bash
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user sussekl@mail.uni-paderborn.de
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=150G
#SBATCH -J "TempME_preprocessing_%a"
#SBATCH -p normal
#SBATCH -A hpc-prf-wiki
#SBATCH --array=
#SBATCH -J "TempME_preprocessing_%A_%a"
#SBATCH -o "Slurm/TempME_preprocessing_%A_%a.out"

module load lang/Python/3.11.5-GCCcore-13.2.0
module load lib/libffi/3.4.4-GCCcore-13.2.0
module load lang/Tkinter/3.11.5-GCCcore-13.2.0
module load system/CUDA/12.4.1
source .tgnn_shap_venv/bin/activate

options=( "Flights" "MOOC" "Reddit" "UNtrade" "UNvote" "USLegis" "Wikipedia" )

echo "Running preprocessing for dataset: ${options[$SLURM_ARRAY_TASK_ID]}"

python -m Evaluation.tempme_preprocessing --dataset ${options[$SLURM_ARRAY_TASK_ID]}