#!/bin/bash
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user sussekl@mail.uni-paderborn.de
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH -p gpu
#SBATCH -A hpc-prf-dedsm
#SBATCH -J "Event_vs_Feature_Explainer_%A_%a"
#SBATCH -o "Slurm/Event_vs_Feature_Explainer_%A_%a.out"
#SBATCH --array=0-6

module load lang/Python/3.11.5-GCCcore-13.2.0
module load lib/libffi/3.4.4-GCCcore-13.2.0
module load lang/Tkinter/3.11.5-GCCcore-13.2.0
module load system/CUDA/12.4.1
source .tgnn_shap_venv/bin/activate

options=( "Flights" "MOOC" "Reddit" "UNtrade" "UNvote" "USLegis" "Wikipedia" )

echo "Running for dataset: ${options[$SLURM_ARRAY_TASK_ID]}"

python -m Evaluation.misc_tests.event_feature_contributions --dataset ${options[$SLURM_ARRAY_TASK_ID]} --n_events 10 --max_num_samples 550
