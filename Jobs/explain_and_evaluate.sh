#!/bin/bash
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user sussekl@mail.uni-paderborn.de
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH -p gpu
#SBATCH -A hpc-prf-dedsm
#SBATCH -J "Explain_and_evaluate_%A_%a"
#SBATCH -o "Slurm/Explain_and_evaluate_%A_%a.out"
#SBATCH --array=1,6


module load lang/Python/3.11.5-GCCcore-13.2.0
module load lib/libffi/3.4.4-GCCcore-13.2.0
module load lang/Tkinter/3.11.5-GCCcore-13.2.0
module load system/CUDA/12.4.1
source .tgnn_shap_venv/bin/activate

datasets=( "Flights" "MOOC" "Reddit" "UNtrade" "UNvote" "USLegis" "Wikipedia" "WikipediaCAWN" "RedditCAWN" )
datasets=( "MOOC" )
explainers=( "shapley_event" "shapley_feature" "tgnn" "tempme" "qiea" "random_event" "random_feature" )

forbidden_combinations=( 
    "tempme:WikipediaCAWN" "tempme:Flights" 
    "tgnn:WikipediaCAWN" "shapley_feature:WikipediaCAWN" 
    "tgnn:RedditCAWN" "shapley_feature:RedditCAWN" 
    "tempme:RedditCAWN" )

dataset=${datasets[$((SLURM_ARRAY_TASK_ID / ${#explainers[@]}))]}
explainer=${explainers[$((SLURM_ARRAY_TASK_ID % ${#explainers[@]}))]}

forbidden_combination="${explainer}:${dataset}"
forbidden="${forbidden_combination}"
if [[ " ${forbidden_combinations[*]} " == *" ${forbidden} "* ]]; then
    echo "Skipping evaluation for dataset: ${dataset} with explainer: ${explainer}"
    exit 0
fi

echo "Running evaluation for dataset: ${dataset} with explainer: ${explainer}"

python -m Evaluation.explain_and_evaluate --action both --num_samples 2 --dataset ${dataset} --explainer ${explainer}
