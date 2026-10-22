#!/bin/bash
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user sussekl@mail.uni-paderborn.de
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH -p gpu
#SBATCH -A hpc-prf-wiki
#SBATCH -J "Explain_and_evaluate_shapley_feature_%A_%a"
#SBATCH -o "Slurm/%A_%a_Explain_and_evaluate_shapley_feature.out"
#SBATCH --array=6-8

module load system/CUDA/12.4.1
module load tools/Apptainer/1.1.9-GCCcore-11.3.0

datasets=( "Flights" "MOOC" "Reddit" "UNtrade" "UNvote" "USLegis" "Wikipedia" "WikipediaCAWN" "RedditCAWN" )
explainers=( "shapley_event" "shapley_feature" "tgnn" "tempme" "qiea" "random_event" "random_feature" )
explainers=( "shapley_feature" )

forbidden_combinations=( 
    "tempme:WikipediaCAWN" "tempme:Flights" 
    "tgnn:WikipediaCAWN"
    "tgnn:RedditCAWN"
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

apptainer exec --nv --contain --cleanenv --bind "$PWD:$PWD" --pwd "$PWD" pytorch26-cu124.sif python -u -m Evaluation.explain_and_evaluate --output_dir ./Results --action evaluate --num_samples 200 --dataset ${dataset} --explainer ${explainer}
