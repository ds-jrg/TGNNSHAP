from argparse import ArgumentParser
import os
parser = ArgumentParser()
parser.add_argument("-d", "--dataset", dest="dataset",
                    help="dataset name", metavar="DATASET", required=True)
parser.add_argument("-num_samples", "-n", dest="num_samples", type=int, default=10_000)

args = parser.parse_args()

from Config.config import CONFIG
CONFIG = CONFIG(args.dataset)

from Evaluation.model import load_model_and_data

import numpy as np
import pandas as pd
from tqdm import tqdm

import time

from Explainers.Shapley4TGNN.Explainer import ShapleyExplainerFeatures

# Use the shared model/data construction used by the main evaluation scripts.
model, full_data, _train_data, _val_data, _test_data, full_neighbor_sampler, _train_sampler, _random_sampler = load_model_and_data()
edge_raw_features = full_neighbor_sampler.edge_features.detach().cpu().numpy()

input_file = f"Results/Feat_explainer_explanations/{args.dataset}/Shapley4TGNNFeature.csv"

if not os.path.exists(input_file):
    raise FileNotFoundError(f"File {input_file} does not exist. ")

output_file = f"Results/Feat_explainer_explanations/{args.dataset}/Shapley4TGNNFeature_high_quality.csv"
if not os.path.exists(output_file):
    with open(output_file, 'w') as f:
        f.write("e_id,explained_e_id,time_taken,model,num_samples,explanations\n")


explanations = pd.read_csv(input_file)
events = explanations.loc[:, ['e_id', 'explained_e_id']].drop_duplicates()


print(f"Number of events to explain: {len(events)}")
print(f"Number of samples: {args.num_samples}")
print(f"Output file: {output_file}")

explainer = ShapleyExplainerFeatures(model, full_neighbor_sampler, full_data, edge_raw_features, feature_names=True, shapley_alg="MonteCarlo")
explainer.initialize()

for e_id, explained_e_id in tqdm(events.values, desc="Computing explanation variance"):
    src = int(full_data.src_node_ids[e_id])
    dst = int(full_data.dst_node_ids[e_id])
    timestamp = int(full_data.node_interact_times[e_id])
    subgraphs_src = full_neighbor_sampler.get_multi_hop_neighbors(CONFIG.model.num_layers, np.array([src]), np.array([timestamp]), num_neighbors = CONFIG.model.num_neighbors)
    subgraphs_dst = full_neighbor_sampler.get_multi_hop_neighbors(CONFIG.model.num_layers, np.array([dst]), np.array([timestamp]), num_neighbors = CONFIG.model.num_neighbors)   
    explanation = explainer.explain_instance(src, dst, timestamp, event_id=explained_e_id, silent=True, max_num_samples=args.num_samples)[0]
    explanation = np.array(sorted(explanation, key=lambda x: (x[0], x[5])))
    row = {"e_id": e_id, 
            "explained_e_id": explained_e_id,
            "model": "monte_carlo",
            "num_samples": args.num_samples,
            "explanations": explanation[:,-1].astype(np.float32)}
    with open(output_file, 'a') as f:
        f.write(f"{row['e_id']},{row['explained_e_id']},{row['model']},{row['num_samples']},{row['explanations'].tolist()}\n")
                    
print("Done generating explanations.")       