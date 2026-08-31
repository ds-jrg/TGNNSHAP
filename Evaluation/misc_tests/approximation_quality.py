from argparse import ArgumentParser
import os
parser = ArgumentParser()
parser.add_argument("-d", "--dataset", dest="dataset",
                    help="dataset name", metavar="DATASET", required=True)
parser.add_argument("--n_events", type=int, default=2,
                    help="number of graph events to explain")
parser.add_argument("--n_explained_events", type=int, default=2,
                    help="number of neighborhood events explained per graph event")
parser.add_argument("--n_reps", type=int, default=2,
                    help="number of repetitions for each sample size and event")
parser.add_argument("--n_sample_sizes", type=int, default=10,
                    help="number of sample sizes to evaluate")
parser.add_argument("--override", action="store_true", default=False,
                    help="delete the existing output file before starting")

args = parser.parse_args()

from Config.config import CONFIG
CONFIG = CONFIG(args.dataset)

from DyGLib.models.GraphMixer import GraphMixer
from DyGLib.models.TGAT import TGAT
from DyGLib.models.TCL import TCL
from DyGLib.models.CAWN import CAWN
from DyGLib.models.DyGFormer import DyGFormer
from DyGLib.models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from DyGLib.models.modules import TGNN, NeuralNetworkSrcDst
from DyGLib.utils.DataLoader import get_link_prediction_data
from DyGLib.utils.utils import get_neighbor_sampler, NegativeEdgeSampler

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import time

from Explainers.Shapley4TGNN.Explainer import ShapleyExplainerFeatures

# # Initialization
trained_model_path = CONFIG.model.trained_model_path
edge_feat_path = CONFIG.data.folder + CONFIG.data.edge_feat_file
node_feat_path = CONFIG.data.folder + CONFIG.data.node_feat_file
index_path = CONFIG.data.folder + CONFIG.data.index_file
feature_names_path = CONFIG.data.folder + CONFIG.data.feature_names_file

# get data for training, validation and testing
node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data = \
    get_link_prediction_data(val_ratio=0.1, test_ratio=0.1, node_dim=CONFIG.model.node_dim)

# initialize validation and test neighbor sampler to retrieve temporal graph
full_neighbor_sampler = get_neighbor_sampler(data=full_data, edge_features=edge_raw_features, sample_neighbor_strategy=CONFIG.model.sample_neighbor_strategy,
                                                time_scaling_factor=CONFIG.model.time_scaling_factor, seed=1)
train_neighbor_sampler = get_neighbor_sampler(data=train_data, edge_features=edge_raw_features, sample_neighbor_strategy=CONFIG.model.sample_neighbor_strategy,
                                                time_scaling_factor=CONFIG.model.time_scaling_factor, seed=1)
full_random_sampler = NegativeEdgeSampler(full_data.src_node_ids, full_data.dst_node_ids, full_data.node_interact_times)

# create model
if CONFIG.model.model_name == 'TGAT':
    dynamic_backbone = TGAT(num_nodes=node_raw_features.shape[0], node_dim=node_raw_features.shape[1], edge_dim=edge_raw_features.shape[1],
                            time_feat_dim=CONFIG.model.time_feat_dim, num_layers=CONFIG.model.num_layers, num_heads=CONFIG.model.num_heads, dropout=CONFIG.model.dropout, device=CONFIG.model.device)
elif CONFIG.model.model_name in ['JODIE', 'DyRep', 'TGN']:
    # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
    src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
        compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
    dynamic_backbone = MemoryModel(num_nodes=node_raw_features.shape[0], node_dim=node_raw_features.shape[1], edge_dim=edge_raw_features.shape[1],
                                    time_feat_dim=CONFIG.model.time_feat_dim, model_name=CONFIG.model.model_name, num_layers=CONFIG.model.num_layers, num_heads=CONFIG.model.num_heads,
                                    dropout=CONFIG.model.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                                    dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=CONFIG.model.device)
elif CONFIG.model.model_name == 'CAWN':
    dynamic_backbone = CAWN(num_nodes=node_raw_features.shape[0], node_dim=node_raw_features.shape[1], edge_dim=edge_raw_features.shape[1],
                            time_feat_dim=CONFIG.model.time_feat_dim, position_feat_dim=CONFIG.model.position_feat_dim, walk_length=CONFIG.model.walk_length,
                            num_walk_heads=CONFIG.model.num_walk_heads, dropout=CONFIG.model.dropout, device=CONFIG.model.device)
elif CONFIG.model.model_name == 'TCL':
    dynamic_backbone = TCL(num_nodes=node_raw_features.shape[0], node_dim=node_raw_features.shape[1], edge_dim=edge_raw_features.shape[1],
                            time_feat_dim=CONFIG.model.time_feat_dim, num_layers=CONFIG.model.num_layers, num_heads=CONFIG.model.num_heads,
                            num_depths=CONFIG.model.num_neighbors + 1, dropout=CONFIG.model.dropout, device=CONFIG.model.device)
elif CONFIG.model.model_name == 'GraphMixer':
    dynamic_backbone = GraphMixer(num_nodes=node_raw_features.shape[0], node_dim=node_raw_features.shape[1], edge_dim=edge_raw_features.shape[1],
                            time_feat_dim=CONFIG.model.time_feat_dim, num_tokens=CONFIG.model.num_neighbors, num_layers=CONFIG.model.num_layers, dropout=CONFIG.model.dropout, device=CONFIG.model.device)
elif CONFIG.model.model_name == 'DyGFormer':
    dynamic_backbone = DyGFormer(num_nodes=node_raw_features.shape[0], node_dim=node_raw_features.shape[1], edge_dim=edge_raw_features.shape[1],
                                    time_feat_dim=CONFIG.model.time_feat_dim, channel_embedding_dim=CONFIG.model.channel_embedding_dim, patch_size=CONFIG.model.patch_size,
                                    num_layers=CONFIG.model.num_layers, num_heads=CONFIG.model.num_heads, dropout=CONFIG.model.dropout,
                                    max_input_sequence_length=CONFIG.model.max_input_sequence_length, device=CONFIG.model.device)
else:
    raise ValueError(f"Wrong value for model_name {CONFIG.model.model_name}!")

regressor = NeuralNetworkSrcDst(input_dim=node_raw_features.shape[1], num_layers=CONFIG.model.num_reg_layers, hidden_dim=CONFIG.model.hidden_reg_layers_dim)
model = TGNN(dynamic_backbone, regressor)

model.load_state_dict(torch.load(trained_model_path, weights_only=True, map_location=CONFIG.model.device))
model.to(CONFIG.model.device)
model.eval()


n_events = args.n_events
n_explained_events = args.n_explained_events
n_reps = args.n_reps
n_sample_sizes = args.n_sample_sizes

file = f"Generated_explanations/{args.dataset}/Shapley4TGNNFeature.csv"
if not os.path.exists(os.path.dirname(file)):
    os.makedirs(os.path.dirname(file))
if args.override and os.path.exists(file):
    os.remove(file)
if not os.path.exists(file):
    with open(file, 'w') as f:
        f.write("e_id,explained_e_id,rep,time_taken,model,num_samples,explanations\n")

print("Configuration:")
print(f"  - Number of events to explain: {n_events}")
print(f"  - Number of explained events: {n_explained_events}")
print(f"  - Number of repetitions: {n_reps}")
print(f"  - Number of sample sizes: {n_sample_sizes}")

print(f"{n_events * n_explained_events * n_reps * n_sample_sizes} total explanations will be generated.")
print(f"Output file: {file}")
print(f"{edge_raw_features.shape[1] + 2} many features per event")

max_sample_sizes = min(2**(edge_raw_features.shape[1] + 2), 5_000)
sample_sizes = np.linspace(5, max_sample_sizes, num=n_sample_sizes, dtype=int)
print(f"Sample sizes: {sample_sizes}")

np.random.seed(42)
events_to_explain = np.where(~np.isnan(full_data.labels))[0]
events_to_explain = np.random.choice(events_to_explain, size=n_events, replace=False)

results = []

approx_explainer = ShapleyExplainerFeatures(model, full_neighbor_sampler, full_data, edge_raw_features, feature_names=True, shapley_alg="MonteCarlo")
approx_explainer.initialize()

for e_id in tqdm(events_to_explain, desc="Explaining events"):
    src = int(full_data.src_node_ids[e_id])
    dst = int(full_data.dst_node_ids[e_id])
    timestamp = int(full_data.node_interact_times[e_id])
    subgraphs_src = full_neighbor_sampler.get_multi_hop_neighbors(CONFIG.model.num_layers, np.array([src]), np.array([timestamp]), num_neighbors = CONFIG.model.num_neighbors)
    subgraphs_dst = full_neighbor_sampler.get_multi_hop_neighbors(CONFIG.model.num_layers, np.array([dst]), np.array([timestamp]), num_neighbors = CONFIG.model.num_neighbors)
    all_events = np.concatenate(subgraphs_src[1] + subgraphs_dst[1], axis=None)
    all_events = all_events[all_events != 0]
    inner_events = np.random.choice(all_events, size=n_explained_events, replace=True)
    
    for s in tqdm(sample_sizes, desc="Samples sizes"):
        for e in tqdm(inner_events, desc="Explained events"):
            for r in tqdm(range(n_reps), desc="Repetitions"):
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    # Approximate Owen values
                    start_time = time.time()
                    approx_explanation = approx_explainer.explain_instance(src, dst, timestamp, event_id=e, silent=True, max_num_samples=s)[0]
                    end_time = time.time()
                    approx_explanation = np.array(sorted(approx_explanation, key=lambda x: (x[0], x[5])))
                    row = {"e_id": e_id, 
                           "explained_e_id": e,
                           "rep": r,
                           "time_taken": end_time - start_time,
                           "model": "monte_carlo",
                           "num_samples": s,
                           "explanations": approx_explanation[:,-1].astype(np.float32)}
                    with open(file, 'a') as f:
                        f.write(f"{row['e_id']},{row['explained_e_id']},{row['rep']},{row['time_taken']},{row['model']},{row['num_samples']},{row['explanations'].tolist()}\n")
                    results.append(row)
                    
print("Done generating explanations.")       