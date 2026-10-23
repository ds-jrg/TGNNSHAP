# TGNNSHAP: A Hierarchical Explainer Framework for Temporal Graph Predictions using Shapley and Owen Values

This repository contains the code for the paper *TGNNSHAP: A Hierarchical Explainer Framework for Temporal
Graph Predictions using Shapley and Owen Values*. It trains temporal graph neural networks (TGNNs), creates event- and feature-level explanations, and evaluates the explanations against several baselines. We share intermediate results via a [Google Drive]() and upload them to Zenodo once the paper got accepted.

## Repository layout

- [`Config/`](Config): dataset and model configurations. The available configuration names are `Flights`, `MOOC`, `Reddit`, `RedditCAWN`, `UNtrade`, `UNvote`, `USLegis`, `Wikipedia`, and `WikipediaCAWN`.
- [`Data/`](Data): datasets. Each configured dataset folder contains the `ml_*.csv`, `ml_*.npy`, and `ml_*_node.npy` files referenced by its YAML configuration.
- [`Evaluation/Training/`](Evaluation/Training): link-prediction and link-regression training entry points.
- [`Evaluation/`](Evaluation): explanation generation, evaluation, TempME preprocessing, and visualization.
- [`Explainers/Shapley4TGNN/`](Explainers/Shapley4TGNN): the event- and feature-level Shapley explainers.
- [`Explainers/External/`](Explainers/External): adapted copies of external explainers, including TGNNExplainer, TempME, and QIEA-TGX.
- [`Jobs/`](Jobs): Slurm scripts used for training, preprocessing, explanation generation, and evaluation.
- [`Saved_models/`](Saved_models): trained model checkpoints and training metrics.
- [`Results/`](Results): generated explanations, timing files, and explanation-evaluation results.

## Environment

The project is intended for a CUDA-enabled Linux environment. Install the Python dependencies from [`requirements.txt`](requirements.txt):

```bash
pip install -r requirements.txt
```

Install the CUDA-compatible PyTorch and PyTorch Geometric packages required by the selected environment. The checked-in Slurm jobs use CUDA 12.4 and load Python 3.9 or 3.11 modules; update the module and virtual-environment paths in [`Jobs/`](Jobs) to match the local cluster.

All commands below are run from the repository root. Before running a command, activate the same virtual environment in which the requirements were installed. Alternatively, we provide a container recipe in [`environment.def`](environment.def), which you can use to build a container including all relevant specifications. 

## Datasets and configurations

The YAML file name is the dataset argument passed to every current entry point. For example, `-d MOOC` loads [`Config/MOOC.yaml`](Config/MOOC.yaml). The configuration loader and the dataclass definitions are in [`Config/config.py`](Config/config.py).

The public datasets can be downloaded from [Zenodo](https://zenodo.org/records/7213796). Place each downloaded dataset in the folder specified by its configuration. For example, the MOOC configuration expects:

```text
Data/mooc/ml_mooc.csv
Data/mooc/ml_mooc.npy
Data/mooc/ml_mooc_node.npy
```

The original dataset CSV, when supplied, is optional for the training and evaluation pipeline. Check the corresponding YAML file for the exact file names and folder casing before starting a run.

## Training

The training commands are:

```bash
# Link prediction / classification
python -m Evaluation.Training.prediction -d MOOC

# Link regression
python -m Evaluation.Training.regression -d Flights
```

The task and model are selected by the configuration. Trained checkpoints and metrics are written below [`Saved_models/`](Saved_models), using the dataset and model names configured in the YAML file. Train the model before creating explanations, and ensure that `ModelConfig.trained_model_path` points to the checkpoint to be explained.

For cluster runs, [`Jobs/training_pred.sh`](Jobs/training_pred.sh) submits prediction training as a dataset array. [`Jobs/training_reg.sh`](Jobs/training_reg.sh) contains the regression example.

## TempME preprocessing

TempME requires preprocessing before its explainer can be used. Run it for one dataset with:

```bash
python -m Evaluation.tempme_preprocessing --dataset MOOC
```

The regular dataset array is defined in [`Jobs/tempme_preprocessing.sh`](Jobs/tempme_preprocessing.sh). [`Jobs/tempme_preprocessing_huge_ram.sh`](Jobs/tempme_preprocessing_huge_ram.sh) is provided for the higher-memory `UNtrade` and `UNvote` runs.

## Generate and evaluate explanations

The current pipeline is [`Evaluation/explain_and_evaluate.py`](Evaluation/explain_and_evaluate.py). It supports the following explainer names:

`shapley_event`, `shapley_event_positive`, `shapley_feature`, `tgnn`, `tempme`, `qiea`, `random_event`, and `random_feature`.

Create explanations and evaluate them in one run:

```bash
python -m Evaluation.explain_and_evaluate \
  --dataset MOOC \
  --explainer shapley_event \
  --output_dir Results \
  --action both \
  --num_samples 200
```

The `--action` option can also be `create` or `evaluate`. Use `create` to generate missing explanation files, then use `evaluate` to evaluate files already stored in `Results/Explanations/`. `--use_default_events` enables the default-event replacement used by the evaluation code.

The output directory has this structure:

```text
Results/
├── Explanations/<dataset>/<explainer>/       # compressed .npz explanations
└── Evaluation/<dataset>/                     # timing and evaluation CSV files
```

[`Jobs/explain_and_evaluate.sh`](Jobs/explain_and_evaluate.sh) is the reference Slurm array job. It currently demonstrates a Shapley event explanation run across the configured datasets; change its `explainers` array when running another method. The script also documents combinations that are not supported by the current external explainers.

The feature-explainer approximation experiment is available as [`Jobs/feat_explainer.sh`](Jobs/feat_explainer.sh), which runs [`Evaluation/misc_tests/approximation_quality.py`](Evaluation/misc_tests/approximation_quality.py).

## Visualization

Open [`Evaluation/visualize.ipynb`](Evaluation/visualize.ipynb) to plot the evaluation results. Figures are written below [`Documents/Images/`](Documents/Images/).

## Notes for Slurm users

The scripts in [`Jobs/`](Jobs) assume that:

1. the working directory is the repository root;
2. the required Python, CUDA, and Tkinter modules are available;
3. the virtual environment path in the script exists; and
4. the Slurm account, partition, GPU type, memory, and output paths are valid for the cluster.

Submit a job with the cluster's normal `sbatch Jobs/<script>.sh` workflow after adjusting these values. For local execution, use the Python commands above instead of the Slurm wrappers.



