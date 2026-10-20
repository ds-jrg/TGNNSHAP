#!/bin/bash
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user sussekl@mail.uni-paderborn.de
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH -p gpu
#SBATCH -A hpc-prf-dedsm
#SBATCH -J "Feat_explainer_%A_%a"
#SBATCH -o "Slurm/Feat_explainer_%A_%a.out"
#SBATCH --array=2,6

module load system/CUDA/12.4.1
module load tools/Apptainer/1.1.9-GCCcore-11.3.0

options=( "Flights" "MOOC" "Reddit" "UNtrade" "UNvote" "USLegis" "Wikipedia" )

echo "Running for dataset: ${options[$SLURM_ARRAY_TASK_ID]}"

apptainer exec --nv --contain --cleanenv --bind "$PWD:$PWD" --pwd "$PWD" pytorch26-cu124.sif python -m Evaluation.misc_tests.approximation_quality --dataset ${options[$SLURM_ARRAY_TASK_ID]} 
