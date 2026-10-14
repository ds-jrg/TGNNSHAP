from argparse import ArgumentParser
import os
parser = ArgumentParser()
parser.add_argument("-d", "--dataset", dest="dataset",
                    help="dataset name", metavar="DATASET", required=True)

args = parser.parse_args()

from Config.config import CONFIG
CONFIG = CONFIG(args.dataset)

from Evaluation.model import load_model_and_data

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import time

from Explainers.Shapley4TGNN.Explainer import ShapleyExplainerFeatures, ShapleyExplainerEvents
from Explainers.Shapley4TGNN.Plots import waterfall

# Use the shared model/data construction used by the main evaluation scripts.
model, full_data, _train_data, _val_data, _test_data, full_neighbor_sampler, _train_sampler, _random_sampler = load_model_and_data()
edge_raw_features = full_neighbor_sampler.edge_features.detach().cpu().numpy()

explainer_feature = ShapleyExplainerFeatures(model, full_neighbor_sampler, full_data, edge_raw_features, top_k=3, shapley_alg="MonteCarlo", feature_names=True)
explainer_feature.initialize()

num_samples = 20
np.random.seed(42)

sampled_idx = np.random.choice(range(len(full_data.edge_ids)), size=num_samples, replace=False)
sampled_idx = sampled_idx[sampled_idx != 0] 

for e_id in tqdm(sampled_idx, desc="Computing feature explanations"):
    src = int(full_data.src_node_ids[e_id])
    dst = int(full_data.dst_node_ids[e_id])
    timestamp = float(full_data.node_interact_times[e_id])
    explanation_flatten, remaining_ids, remaining_shapley_values, baseline = explainer_feature.explain_instance(src, dst, timestamp, silent=False)
    explanation_flatten = np.array(explanation_flatten)
    not_timing_or_structure = (explanation_flatten[:, -3] != "Structure") & (explanation_flatten[:, -3] != "Timing")
    explanation_flatten[not_timing_or_structure, -3] = "Feat. " + explanation_flatten[not_timing_or_structure, -3] 
    print(explanation_flatten[:,-2])
    explanation_flatten[:,-2] = np.round(explanation_flatten[:,-2].astype(float), 2)
    fig = waterfall(explanation_flatten, baseline, 4, 
                src=str(src), dst=str(dst), ts=str(timestamp), target=str(1), 
                remaining_ids=remaining_ids,remaining_shapley_values=remaining_shapley_values)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["pdf.fonttype"] = 42
    fig.savefig(f"Documents/Images/{args.dataset}/Waterfall_feat_{src}_{dst}_{timestamp}.pdf", bbox_inches="tight")
    print(f"Documents/Images/{args.dataset}/Waterfall_feat_{src}_{dst}_{timestamp}.pdf")

