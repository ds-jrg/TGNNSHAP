from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-d", "--dataset", dest="dataset",
                    help="dataset name", metavar="DATASET", required=True)
parser.add_argument("--explainer", dest="explainer",
                    help="explainer to use", metavar="EXPLAINER", required=True)
parser.add_argument("--num_samples", dest="num_samples", required=False, type=int, default=200,
                    help="number of samples to use for evaluation")
parser.add_argument("--store_coalitions", dest="store_coalitions", required=False, type=bool, default=False,
                    help="whether to store coalitions during evaluation")

args = parser.parse_args()

from Config.config import CONFIG
CONFIG = CONFIG(args.dataset)


from DyGLib.models.GraphMixer import GraphMixer
from DyGLib.models.TGAT import TGAT
from DyGLib.models.TCL import TCL
from DyGLib.models.CAWN import CAWN
from DyGLib.models.DyGFormer import DyGFormer
from DyGLib.models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts

from DyGLib.models.modules import TGNN, MultiHeadAttention, NeuralNetworkSrcDst, BatchSubgraphs
from DyGLib.utils.DataLoader import get_link_prediction_data
from DyGLib.utils.utils import get_neighbor_sampler, NegativeEdgeSampler

from Explainers.utils import Explainer

import torch
import numpy as np
import pandas as pd
import seaborn as sns

import random
import os

from IPython.display import SVG
import time

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

model.load_state_dict(torch.load(trained_model_path, weights_only=True))
model.to(CONFIG.model.device)
model.eval()

num_samples = args.num_samples

def get_edge_by_id(link_index):
    src, dst, time_stamp, edge_id = full_data.src_node_ids[link_index], full_data.dst_node_ids[link_index], full_data.node_interact_times[link_index], full_data.edge_ids[link_index]
    if CONFIG.model.task == "regression":
        true_value = full_data.labels[link_index]
    else: # link prediction case
        true_value = 1
    return src, dst, time_stamp, edge_id, true_value

random.seed(2025)
sampled_edge_ids = random.sample((np.where((~np.isnan(full_data.labels)) & (~np.isin(full_data.edge_ids, train_data.edge_ids)))[0]).tolist(), num_samples)
edge_info_array = np.array([list(get_edge_by_id(i)) for i in sampled_edge_ids])
edge_info = pd.DataFrame(edge_info_array, columns=["Src", "Dst", "Time", "Event", "Target"])
edges = edge_info["Event"].to_numpy(dtype=int)
edge_info["InTrain"] = np.isin(edges, train_data.edge_ids)
edge_info = edge_info.sort_values(by="InTrain").reset_index(drop=True)
edge_info = edge_info[edge_info.InTrain == False]
srcs = edge_info["Src"].to_numpy(dtype=int)
dsts = edge_info["Dst"].to_numpy(dtype=int)
timestamps = edge_info["Time"].to_numpy(dtype="float64")
targets = edge_info["Target"].to_numpy(dtype="float64")

model.eval()

subgraphs_src = full_neighbor_sampler.get_multi_hop_neighbors(CONFIG.model.num_layers, srcs, timestamps, num_neighbors = CONFIG.model.num_neighbors)
subgraphs_dst = full_neighbor_sampler.get_multi_hop_neighbors(CONFIG.model.num_layers, dsts, timestamps, num_neighbors = CONFIG.model.num_neighbors)
edge_feat_src = full_neighbor_sampler.get_edge_features_for_multi_hop(subgraphs_src[1])
edge_feat_dst = full_neighbor_sampler.get_edge_features_for_multi_hop(subgraphs_dst[1])

subgraphs_src = BatchSubgraphs(*subgraphs_src, edge_feat_src)
subgraphs_src.to(CONFIG.model.device)
subgraphs_dst = BatchSubgraphs(*subgraphs_dst, edge_feat_dst)
subgraphs_dst.to(CONFIG.model.device)

predicts = model(src_node_ids=srcs,
                dst_node_ids=dsts,
                node_interact_times=timestamps,
                src_subgraphs = subgraphs_src,
                dst_subgraphs = subgraphs_dst,
                time_gap=CONFIG.model.time_gap,
                edges_are_positive=True).squeeze(dim=-1).sigmoid()

edge_info["Prediction"] = predicts.detach().cpu().numpy()

# normalize explainer selection
_selected = [args.explainer]
# map aliases
_alias_map = {
    "shapley4tgnnevent": "shapley_event",
    "shapley_event": "shapley_event",
    "shapleyfeature": "shapley_feature",
    "shapley_feature": "shapley_feature",
    "feature": "shapley_feature",
    "tgnn": "tgnn",
    "tgnnexplainer": "tgnn",
    "tempme": "tempme",
    "random": "random",
    "randomexplainer": "random",
    "baseline": "random",
    "all": "all"
}
selected = set()
for s in _selected:
    mapped = _alias_map.get(s)
    if mapped:
        selected.add(mapped)
    else:
        raise ValueError(f"Unknown explainer '{s}'. Allowed: shapley_event, shapley_feature, tgnn, tempme, random, all")

if "all" in selected:
    selected = {"shapley_event", "shapley_feature", "tgnn", "tempme", "random"}

results_list = []
timings_list = []


def evaluate_explainer(explainer: Explainer, explainer_name):
    start = time.time_ns()
    explainer.initialize()
    end = time.time_ns()

    intermediate_results_path = (
        f"Results/{CONFIG.data.dataset_name}/{explainer_name}.csv"
    )
    timings_path = (
        f"Results/{CONFIG.data.dataset_name}/{explainer_name}_timings.csv"
    )
    timing_columns = ["Time(ns)", "Time(s)", "Explainer", "Stage", "Instance Index"]
    os.makedirs(os.path.dirname(timings_path), exist_ok=True)
    pd.DataFrame([{
        "Time(ns)": end - start,
        "Time(s)": (end - start) / 1_000_000_000,
        "Explainer": explainer_name,
        "Stage": "Init",
        "Instance Index": None,
    }], columns=timing_columns).to_csv(timings_path, index=False)

    results, exec_times = explainer.evaluate(
        srcs,
        dsts,
        timestamps,
        targets,
        edge_raw_features,
        store_coalitions=args.store_coalitions,
        intermediate_results_path=intermediate_results_path,
        timings_path=timings_path,
        explainer_name=explainer_name,
    )
    results["Explainer"] = explainer_name
    aggregated_results_path = (
        f"Results/{CONFIG.data.dataset_name}/{explainer_name}_agg.csv"
    )
    os.makedirs(os.path.dirname(aggregated_results_path), exist_ok=True)
    results.to_csv(aggregated_results_path, index=False)
    
    timings = pd.read_csv(timings_path)
    return results, timings
    
    
if "shapley_event" in selected:
    from Explainers.Shapley4TGNN.Explainer import ShapleyExplainerEvents
    
    print("Evaluating Shapley4TGNNEvent...")
    
    explainer = ShapleyExplainerEvents(model, full_neighbor_sampler, full_data, edge_raw_features)
    results, timings = evaluate_explainer(explainer, "Shapley4TGNNEvent")
    
    results_list.append(results)
    timings_list.append(timings)
    
    explainer = None
    torch.cuda.empty_cache()
    print("Done.")
        
# ## Shapley values - Feature level
if "shapley_feature" in selected:
    from Explainers.Shapley4TGNN.Explainer import ShapleyExplainerFeatures

    print("Evaluating Shapley4TGNNFeature...")

    explainer = ShapleyExplainerFeatures(model, full_neighbor_sampler, full_data, edge_raw_features, None, shapley_alg="MonteCarlo", top_k=3) 
    results, timings = evaluate_explainer(explainer, "Shapley4TGNNFeature")
    
    results_list.append(results)
    timings_list.append(timings)
    
    explainer = None
    torch.cuda.empty_cache()
    print("Done.")

# ## TGNN Explainer
if "tgnn" in selected:
    from Explainers.External.tgnnexplainer.Explainer import SubgraphXTExplainer
    
    print("Evaluating TGNNExplainer...")
    
    # Make sure the model is TGAT and set the edge_attention_alter_mode to "add" for all MultiHeadAttention layers
    assert isinstance(model.backbone, TGAT), "Model must be an instance of TGAT if using TGNNExplainer."
    for layer in model.backbone.temporal_conv_layers:
        assert isinstance(layer, MultiHeadAttention), "Layer must be an instance of MultiHeadAttention"
        layer.edge_attention_alter_mode = "add"
    
    SubgraphXTExplainer.train_model_if_missing(
        model, full_neighbor_sampler, full_data #Training set is constructed within the explainer
    )
    explainer = SubgraphXTExplainer(model, full_neighbor_sampler, full_data)
    results, timings = evaluate_explainer(explainer, "TGNNExplainer")
    
    results_list.append(results)
    timings_list.append(timings)
    
    explainer = None
    torch.cuda.empty_cache()
    print("Done.")
    
    

# ## TempME
if "tempme" in selected:
    from Explainers.External.TempME.Explainer import TempMEExplainer
    
    print("Evaluating TempME...")
    
    # Make sure the model is TGAT and set the edge_attention_alter_mode to "add" for all MultiHeadAttention layers
    assert isinstance(model.backbone, TGAT), "Model must be an instance of TGAT if using TGNNExplainer."
    for layer in model.backbone.temporal_conv_layers:
        assert isinstance(layer, MultiHeadAttention), "Layer must be an instance of MultiHeadAttention"
        layer.edge_attention_alter_mode = "multiply"

    # train_model_if_missing() ensures training data exists before training;
    # initialize() only loads cached artifacts and raises if they are absent.
    TempMEExplainer.preprocess_data_if_missing(train_data, subset_name="train")
    TempMEExplainer.train_model_if_missing(
        model, train_neighbor_sampler, full_neighbor_sampler, full_random_sampler, train_data, full_data, CONFIG.model.device
    )

    TempMEExplainer.preprocess_data_if_missing(full_data, subset_name="test")
    explainer = TempMEExplainer(model, full_neighbor_sampler, full_data)

    results, timings = evaluate_explainer(explainer, "TempME")
      
    results_list.append(results)
    timings_list.append(timings)
    
    explainer = None
    torch.cuda.empty_cache()
    print("Done.")


# ## Random baseline
if "random" in selected:
    from Explainers.RandomExplainer.Explainer import RandomExplainer

    print("Evaluating Random baseline...")

    explainer = RandomExplainer(
        model, full_neighbor_sampler, full_data, edge_raw_features
    )
    results, timings = evaluate_explainer(explainer, "Random")

    results_list.append(results)
    timings_list.append(timings)

    explainer = None
    torch.cuda.empty_cache()
    print("Done.")
