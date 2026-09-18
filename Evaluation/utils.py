

from copy import deepcopy

import torch

from Config.config import CONFIG
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
CONFIG = CONFIG()

from DyGLib.utils.utils import BatchSubgraphs, NeighborSampler
from Explainers.Shapley4TGNN.Explainer import ShapleyExplainerEvents



def get_shapley_value(event_explainer:ShapleyExplainerEvents, src, dst, timestamp, event_id, sg_src=None, sg_dst=None):
    event_ids, shap_values = event_explainer.explain_instance(src, dst, timestamp, silent=True, subgraphs_src=sg_src, subgraphs_dst=sg_dst)
    shap_value_event = shap_values.values[0][event_ids == event_id]
    if len(shap_value_event) != 1:
        raise ValueError(f"Expected one Shapley value for event {event_id}, but got {len(shap_value_event)}")
    return shap_value_event[0]

def get_shapley_value_of_masked_sg(features,src, dst, timestamp, event_id, event_features, event_explainer:ShapleyExplainerEvents, sg_src: BatchSubgraphs, sg_dst: BatchSubgraphs):
    sg_src_pos = deepcopy(sg_src)
    sg_dst_pos = deepcopy(sg_dst)
    if 1 in features:
        sg_src_pos.mask_event_timing(event_id, 0)
        sg_dst_pos.mask_event_timing(event_id, 0)
    if 0 in features:
        sg_src_pos.mask_node_attention(event_id, torch.tensor(0.0))
        sg_dst_pos.mask_node_attention(event_id, torch.tensor(0.0))
    features = features - 2
    features = features[features >= 0]
    masked_features = event_features
    masked_features[features] = 0.0
    sg_src_pos.mask_event_features(event_id, torch.tensor(masked_features))
    sg_dst_pos.mask_event_features(event_id, torch.tensor(masked_features))
    
    shapley = get_shapley_value(event_explainer, src, dst, timestamp, event_id, sg_src=sg_src_pos, sg_dst=sg_dst_pos)
    return shapley


def evaluate_file(file_name, directory, random_directory, neighbor_finder: NeighborSampler, data, event_explainer, sparsity_thresholds):
    k = 1
    file_path = os.path.join(directory, file_name)
    random_file_path = os.path.join(random_directory, file_name)
    
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Explanation file not found: {file_path}")
    if not os.path.isfile(random_file_path):
        raise FileNotFoundError(f"Random explanation file not found: {random_file_path}")
    
    explanations = np.load(file_path, allow_pickle=True)["explanations"]
    top_k_events = explanations[np.sort(np.unique(explanations[:, 0], return_index=True)[1])[:k], :]
    random_explanations = np.load(random_file_path, allow_pickle=True)["explanations"]
    top_k_random_events = random_explanations[np.sort(np.unique(random_explanations[:, 0], return_index=True)[1])[:k], :]
    
    results = []
    print(f"Evaluating file: {file_name} with top {k} events.")
    
    src, _, dst, timestamp = file_name.replace(".npz", "").split("_")
    src = int(src)
    dst = int(dst)
    timestamp = float(timestamp)
    
    for e_id, random_e_id in zip(top_k_events[:, 0], top_k_random_events[:, 0]):
        
        feature_ids = explanations[explanations[:, 0] == e_id, 1]
        feature_ids_random = random_explanations[random_explanations[:, 0] == random_e_id, 1]
        
        assert np.array_equal(np.sort(feature_ids), np.sort(feature_ids_random)), \
            f"Feature IDs do not match for event {e_id} and random event {random_e_id}."
        
        shapley_value = get_shapley_value(event_explainer, src, dst, timestamp, e_id)
        
        subgraphs_src = neighbor_finder.get_multi_hop_neighbors(CONFIG.model.num_layers, np.array([src]), np.array([timestamp]), num_neighbors=CONFIG.model.num_neighbors)
        event_feat_src = neighbor_finder.get_edge_features_for_multi_hop(subgraphs_src[1])
        subgraphs_src = BatchSubgraphs(*subgraphs_src, event_feat_src)
        
        subgraphs_dst = neighbor_finder.get_multi_hop_neighbors(CONFIG.model.num_layers, np.array([dst]), np.array([timestamp]), num_neighbors=CONFIG.model.num_neighbors)
        event_feat_dst = neighbor_finder.get_edge_features_for_multi_hop(subgraphs_dst[1])
        subgraphs_dst = BatchSubgraphs(*subgraphs_dst, event_feat_dst)
        
        event_features = neighbor_finder.edge_features[e_id].detach().cpu().numpy()
        
        for s in tqdm(sparsity_thresholds):
            i = int(len(feature_ids) * s)
            features = feature_ids[:i]
            features_random = feature_ids_random[:i]
            
            features_neg = feature_ids[i:]
            features_random_neg = feature_ids_random[i:]
            
            shapley_pos = get_shapley_value_of_masked_sg(features, src, dst, timestamp, e_id, event_features, event_explainer, subgraphs_src, subgraphs_dst)
            shapley_pos_random = get_shapley_value_of_masked_sg(features_random, src, dst, timestamp, random_e_id, event_features, event_explainer, subgraphs_src, subgraphs_dst)
            shapley_neg = get_shapley_value_of_masked_sg(features_neg, src, dst, timestamp, e_id, event_features, event_explainer, subgraphs_src, subgraphs_dst)
            shapley_neg_random = get_shapley_value_of_masked_sg(features_random_neg, src, dst, timestamp, random_e_id, event_features, event_explainer, subgraphs_src, subgraphs_dst)
            
            results.append({
                "src": src, "dst": dst, "timestamp": timestamp,
                "ground_truth": shapley_value,
                "shapley_value_pos": shapley_pos,
                "shapley_value_neg": shapley_neg,
                "shapley_value_pos_random": shapley_pos_random,
                "shapley_value_neg_random": shapley_neg_random,
                "sparsity_thresholds": s,
            })
            
    results = pd.DataFrame(results)
       
    return results

def evaluate_feature_explanations(model, sampler, data, event_features, sparsity_thresholds) -> None:
    explainer_name = "Shapley4TGNNFeature"
    random_explainer_name = "RandomFeature"
    
    directory = os.path.join("Results", "Explanations", CONFIG.data.dataset_name, explainer_name)
    random_directory = os.path.join("Results", "Explanations", CONFIG.data.dataset_name, random_explainer_name)
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Explanation directory not found: {directory}")
    if not os.path.isdir(random_directory):
        raise FileNotFoundError(f"Random explanation directory not found: {random_directory}")
    
    files = sorted(file for file in os.listdir(directory) if file.endswith(".npz"))
    if not files:
        raise FileNotFoundError(f"No explanation files found in {directory}")
    
    event_explainer = ShapleyExplainerEvents(model, sampler, data, event_features)
    event_explainer.initialize()
    
    frames = [evaluate_file(file, directory, random_directory, sampler, data, event_explainer, sparsity_thresholds)
              for file in tqdm(files, desc="Evaluating explanation files", unit="file")]
    output = os.path.join("Results", "Evaluation", CONFIG.data.dataset_name,
                          f"{explainer_name}_explanation_evaluation.csv")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output, index=False)
    print(f"Saved explanation evaluation results to {output}")