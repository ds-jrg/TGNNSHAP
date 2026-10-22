#!/bin/bash
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user sussekl@mail.uni-paderborn.de
#SBATCH -t 3:00:00
#SBATCH --mem=60000
#SBATCH --gres=gpu:1
#SBATCH -p irb-40GB
#SBATCH -J "Event_vs_Feature_Explainer_%A_%a"
#SBATCH -o "Event_vs_Feature_Explainer_%A_%a.out"
#SBATCH --array=0-6

# module load lang/Python/3.11.5-GCCcore-13.2.0
# module load lib/libffi/3.4.4-GCCcore-13.2.0
# module load lang/Tkinter/3.11.5-GCCcore-13.2.0
# module load system/CUDA/12.4.1
source .tgnnshap_venv/bin/activate

options=( "Flights" "MOOC" "Reddit" "UNtrade" "UNvote" "USLegis" "Wikipedia" )

echo "Running for dataset: ${options[$SLURM_ARRAY_TASK_ID]}"

python -m Evaluation.misc_tests.event_feature_contributions --dataset ${options[$SLURM_ARRAY_TASK_ID]} --n_events 30 --max_num_samples 550
