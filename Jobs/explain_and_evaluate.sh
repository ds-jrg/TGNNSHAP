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
#SBATCH --array=5

module load lang/Python/3.11.5-GCCcore-13.2.0
module load lib/libffi/3.4.4-GCCcore-13.2.0
module load lang/Tkinter/3.11.5-GCCcore-13.2.0
module load system/CUDA/12.4.1
source .tgnn_shap_venv/bin/activate

datasets=( "Flights" "MOOC" "Reddit" "UNtrade" "UNvote" "USLegis" "Wikipedia" "WikipediaCAWN" )
explainers=( "shapley_event" "shapley_feature" "tgnn" "tempme" "qiea" "random" )

forbidden_combinations=( "tempme:WikipediaCAWN" "tempme:Flights" "tgnn:WikipediaCAWN" "shapley_feature:WikipediaCAWN" )

dataset=${datasets[$((SLURM_ARRAY_TASK_ID / ${#explainers[@]}))]}
explainer=${explainers[$((SLURM_ARRAY_TASK_ID % ${#explainers[@]}))]}

forbidden_combination="${explainer}:${dataset}"
forbidden="${forbidden_combination}"
if [[ " ${forbidden_combinations[*]} " == *" ${forbidden} "* ]]; then
    echo "Skipping evaluation for dataset: ${dataset} with explainer: ${explainer}"
    exit 0
fi

echo "Running evaluation for dataset: ${dataset} with explainer: ${explainer}"

python -m Evaluation.explain_and_evaluate --both --num_samples 200 --dataset ${dataset} --explainer ${explainer}
