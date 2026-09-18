"""Generate temporal-graph explanations and evaluate them.

The pipeline is controlled by ``--action``:

* ``create``: generate and store explanations and timings;
* ``evaluate``: evaluate already stored explanation files;
* ``both``: perform both steps (the default).

Model construction and data loading are delegated to ``Evaluation.model``.
"""

from __future__ import annotations

from argparse import ArgumentParser
import copy
import os
import random
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, help="dataset name")
parser.add_argument("--explainer", required=True, help="explainer to use")
parser.add_argument(
    "--action", "--mode", dest="action",
    choices=("create", "evaluate", "both"), default="both",
    help="create explanations, evaluate existing explanations, or do both",
)
parser.add_argument("--num_samples", type=int, default=200,
                    help="number of test interactions used when creating explanations")
args = parser.parse_args()

# CONFIG is a singleton. Initialise it before importing modules whose classes
# read the active dataset configuration at import time.
from Config.config import CONFIG

CONFIG = CONFIG(args.dataset)

from Evaluation.model import load_model_and_data
from DyGLib.models.TGAT import TGAT
from DyGLib.models.modules import MultiHeadAttention
from DyGLib.models.modules import BatchSubgraphs
from Explainers.utils import Explainer, to_object_array
from Evaluation.utils import evaluate_feature_explanations


EXPLAINER_ALIASES = {
    "shapley4tgnnevent": "shapley_event",
    "shapley_event": "shapley_event",
    "shapley4tgnneventpositive": "shapley_event_positive",
    "shapley_event_positive": "shapley_event_positive",
    "shapley_event_pos": "shapley_event_positive",
    "shapleyfeature": "shapley_feature",
    "shapley_feature": "shapley_feature",
    "feature": "shapley_feature",
    "tgnn": "tgnn",
    "tgnnexplainer": "tgnn",
    "tempme": "tempme",
    "qiea": "qiea",
    "qieatgx": "qiea",
    "qiea-tgx": "qiea",
    "random": "random_event",
    "random_event": "random_event",
    "randomevent": "random_event",
    "randomexplainer": "random_event",
    "baseline": "random_event",
    "random_feature": "random_feature",
    "randomfeature": "random_feature",
}

EXPLAINER_DIRECTORIES = {
    "shapley_event": "Shapley4TGNNEvent",
    "shapley_event_positive": "Shapley4TGNNEventPositive",
    "shapley_feature": "Shapley4TGNNFeature",
    "tgnn": "TGNNExplainer",
    "tempme": "TempME",
    "qiea": "QIEA-TGX",
    "random_event": "RandomEvent",
    "random_feature": "RandomFeature",
}


def normalize_explainer_name(name: str) -> str:
    try:
        return EXPLAINER_ALIASES[name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown explainer '{name}'. Allowed: "
            + ", ".join(EXPLAINER_DIRECTORIES)
        ) from exc


def select_test_edges(full_data, train_data, num_samples: int):
    candidates = np.where(
        (~np.isnan(full_data.labels))
        & (~np.isin(full_data.edge_ids, train_data.edge_ids))
    )[0].tolist()
    if num_samples < 1 or num_samples > len(candidates):
        raise ValueError(
            f"num_samples must be between 1 and {len(candidates)}, got {num_samples}"
        )
    indices = random.sample(candidates, num_samples)
    return (
        full_data.src_node_ids[indices].astype(int),
        full_data.dst_node_ids[indices].astype(int),
        full_data.node_interact_times[indices].astype(float),
        full_data.labels[indices].astype(float),
    )


def make_explainer(name: str, model, full_sampler, train_sampler, full_data,
                   train_data, full_random_sampler, edge_features) -> Explainer:
    if name == "shapley_event":
        from Explainers.Shapley4TGNN.Explainer import ShapleyExplainerEvents
        return ShapleyExplainerEvents(model, full_sampler, full_data, edge_features)
    if name == "shapley_event_positive":
        from Explainers.Shapley4TGNN.Explainer import ShapleyExplainerEventsPositive
        return ShapleyExplainerEventsPositive(model, full_sampler, full_data, edge_features)
    if name == "shapley_feature":
        from Explainers.Shapley4TGNN.Explainer import ShapleyExplainerFeatures
        return ShapleyExplainerFeatures(
            model, full_sampler, full_data, edge_features, None,
            shapley_alg="MonteCarlo", top_k=1,
        )
    if name in ("random_event", "random_feature"):
        from Explainers.RandomExplainer.Explainer import (
            RandomExplainer,
            RandomFeatureExplainer,
        )
        if name == "random_feature":
            return RandomFeatureExplainer(model, full_sampler, full_data, edge_features, top_k=1)
        return RandomExplainer(model, full_sampler, full_data, edge_features)
    if name == "tgnn":
        from Explainers.External.tgnnexplainer.Explainer import SubgraphXTExplainer
        if not isinstance(model.backbone, TGAT):
            raise AssertionError("TGNNExplainer requires a TGAT backbone")
        for layer in model.backbone.temporal_conv_layers:
            if not isinstance(layer, MultiHeadAttention):
                raise AssertionError("TGAT layers must be MultiHeadAttention")
            layer.edge_attention_alter_mode = "add"
        SubgraphXTExplainer.train_model_if_missing(model, full_sampler, full_data)
        return SubgraphXTExplainer(model, full_sampler, full_data)
    if name == "tempme":
        from Explainers.External.TempME.Explainer import TempMEExplainer
        if not isinstance(model.backbone, TGAT):
            raise AssertionError("TempME requires a TGAT backbone")
        for layer in model.backbone.temporal_conv_layers:
            if not isinstance(layer, MultiHeadAttention):
                raise AssertionError("TGAT layers must be MultiHeadAttention")
            layer.edge_attention_alter_mode = "multiply"
        TempMEExplainer.preprocess_data_if_missing(train_data, subset_name="train")
        TempMEExplainer.train_model_if_missing(
            model, train_sampler, full_sampler, full_random_sampler,
            train_data, full_data, CONFIG.model.device,
        )
        TempMEExplainer.preprocess_data_if_missing(full_data, subset_name="test")
        return TempMEExplainer(model, full_sampler, full_data)
    if name == "qiea":
        from Explainers.External.QIEATGX.Explainer import QIEATGXExplainer
        return QIEATGXExplainer(model, full_sampler, full_data)
    raise AssertionError(f"Unhandled explainer: {name}")


def write_explanation(path: str, explanations, sg_src, sg_dst) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"explanations": explanations}
    if sg_src is not None and sg_dst is not None:
        payload.update(
            sg_src_events=to_object_array(sg_src.events),
            sg_dst_events=to_object_array(sg_dst.events),
            sg_src_timestamps=to_object_array(sg_src.timestamps),
            sg_dst_timestamps=to_object_array(sg_dst.timestamps),
            sg_src_node_ids=to_object_array(sg_src.nodes),
            sg_dst_node_ids=to_object_array(sg_dst.nodes),
        )
    np.savez_compressed(path, **payload)


def create_explanations(name: str, model, full_sampler, train_sampler, full_data,
                      train_data, full_random_sampler, edge_features) -> None:
    np.random.seed(42)
    random.seed(42)
    
    srcs, dsts, timestamps, targets = select_test_edges(
        full_data, train_data, args.num_samples
    )
    explainer = make_explainer(
        name, model, full_sampler, train_sampler, full_data, train_data,
        full_random_sampler, edge_features,
    )

    directory = os.path.join("Results", "Explanations", CONFIG.data.dataset_name,
                             EXPLAINER_DIRECTORIES[name])
    timing_path = os.path.join(
        "Results", "Evaluation", CONFIG.data.dataset_name,
        f"{EXPLAINER_DIRECTORIES[name]}_explanation_timings.csv",
    )
    os.makedirs(os.path.dirname(timing_path), exist_ok=True)
    timing_columns = ["Time(ns)", "Time(s)", "Explainer", "Stage", "Instance Index"]
    pd.DataFrame(columns=timing_columns).to_csv(timing_path, index=False)

    initialize_start = time.time_ns()
    explainer.initialize()
    initialize_elapsed = time.time_ns() - initialize_start
    pd.DataFrame([{
        "Time(ns)": initialize_elapsed,
        "Time(s)": initialize_elapsed / 1_000_000_000,
        "Explainer": EXPLAINER_DIRECTORIES[name],
        "Stage": "Init",
        "Src": None,
        "Dst": None,
        "Timestamp": None,
    }]).to_csv(timing_path, mode="a", header=False, index=False)

    for index, (src, dst, timestamp, _target) in enumerate(
        tqdm(zip(srcs, dsts, timestamps, targets), total=len(srcs),
             desc=f"Creating {EXPLAINER_DIRECTORIES[name]} explanations")
    ):
        start = time.time_ns()
        explanation = explainer.explain_instance(src, dst, timestamp, silent=True)
        explanations, sg_src, sg_dst = explainer.build_coalitions(explanation)
        elapsed = time.time_ns() - start
        pd.DataFrame([{
            "Time(ns)": elapsed,
            "Time(s)": elapsed / 1_000_000_000,
            "Explainer": EXPLAINER_DIRECTORIES[name],
            "Stage": "Explain",
            "Src": src,
            "Dst": dst,
            "Timestamp": timestamp,
        }]).to_csv(timing_path, mode="a", header=False, index=False)
        write_explanation(
            os.path.join(directory, f"{src}_to_{dst}_{timestamp}.npz"),
            explanations, sg_src, sg_dst,
        )

    print(f"Saved explanations to {directory}")
    print(f"Saved explanation timings to {timing_path}")


SPARSITY_THRESHOLDS = np.linspace(0, 1, 50)
SPARSITY_THRESHOLDS = 0.5 * (1 + np.tanh(7 * (SPARSITY_THRESHOLDS - 0.5)))
SPARSITY_THRESHOLDS = np.concatenate(([0.0], SPARSITY_THRESHOLDS, [1.0]))


def get_prediction(model, sampler, srcs, dsts, timestamps, subgraphs_src=None,
                   subgraphs_dst=None):
    if subgraphs_src is None or subgraphs_dst is None:
        src_data = sampler.get_multi_hop_neighbors(
            CONFIG.model.num_layers, srcs, timestamps, CONFIG.model.num_neighbors)
        dst_data = sampler.get_multi_hop_neighbors(
            CONFIG.model.num_layers, dsts, timestamps, CONFIG.model.num_neighbors)
        subgraphs_src = BatchSubgraphs(
            *src_data, event_features=sampler.get_edge_features_for_multi_hop(src_data[1]))
        subgraphs_dst = BatchSubgraphs(
            *dst_data, event_features=sampler.get_edge_features_for_multi_hop(dst_data[1]))
    subgraphs_src.to(CONFIG.model.device)
    subgraphs_dst.to(CONFIG.model.device)
    with torch.no_grad():
        logits = model(
            src_node_ids=srcs, dst_node_ids=dsts,
            node_interact_times=timestamps, src_subgraphs=subgraphs_src,
            dst_subgraphs=subgraphs_dst, time_gap=CONFIG.model.time_gap,
            edges_are_positive=False, num_neighbors=CONFIG.model.num_neighbors,
        ).squeeze(-1)
    logits = logits.detach().cpu().numpy()
    return logits, 1 / (1 + np.exp(-logits)), subgraphs_src, subgraphs_dst


def load_explanation(file_name, folder, model, sampler):
    src_text, _, dst_text, timestamp_text = file_name[:-4].split("_")
    src, dst, timestamp = int(src_text), int(dst_text), float(timestamp_text)
    data = np.load(os.path.join(folder, file_name), allow_pickle=True)
    explanations = data["explanations"] if "explanations" in data.files else data["coalitions"]
    sg_src = sg_dst = None
    if "sg_src_events" in data.files and "sg_dst_events" in data.files:
        sg_src_events, sg_dst_events = data["sg_src_events"].tolist(), data["sg_dst_events"].tolist()
        sg_src = BatchSubgraphs(
            events=sg_src_events, nodes=data["sg_src_node_ids"].tolist(),
            timestamps=data["sg_src_timestamps"].tolist(),
            event_features=sampler.get_edge_features_for_multi_hop(sg_src_events),
        )
        sg_dst = BatchSubgraphs(
            events=sg_dst_events, nodes=data["sg_dst_node_ids"].tolist(),
            timestamps=data["sg_dst_timestamps"].tolist(),
            event_features=sampler.get_edge_features_for_multi_hop(sg_dst_events),
        )
    logits, predicts, sg_src, sg_dst = get_prediction(
        model, sampler, np.array([src]), np.array([dst]), np.array([timestamp]),
        sg_src, sg_dst,
    )
    return src, dst, timestamp, explanations, sg_src, sg_dst, logits, predicts

def prepare_subgraphs(sg_src:BatchSubgraphs, sg_dst:BatchSubgraphs, explanations):
    events = np.unique(np.concatenate([sg_src.get_events(), sg_dst.get_events()], axis=1))
    events = events[events != 0]
    if len(events) == 0:
        raise ValueError("Cannot evaluate an explanation with an empty computational subgraph")
    sg_src.repeat_nodes(len(SPARSITY_THRESHOLDS))
    sg_dst.repeat_nodes(len(SPARSITY_THRESHOLDS))
    sg_src_neg, sg_dst_neg = copy.deepcopy(sg_src), copy.deepcopy(sg_dst)
    pos = np.zeros((len(SPARSITY_THRESHOLDS), explanations.shape[1] + 1), dtype=int)
    neg = np.tile(events, (len(SPARSITY_THRESHOLDS), 1))
    sparsities = ((explanations != 0) & np.isin(explanations, events)).sum(axis=1) / len(events)
    for i, threshold in enumerate(SPARSITY_THRESHOLDS):
        mask = sparsities <= threshold
        if mask.any():
            explanation = explanations[mask][np.argmax(sparsities[mask])]
            pos[i, :-1] = explanation
            neg[i, np.isin(events, explanation)] = 0
    neg = np.concatenate([neg, np.zeros((len(SPARSITY_THRESHOLDS), 1), dtype=int)], axis=1)
    sg_src.keep_events(pos)
    sg_dst.keep_events(pos)
    sg_src_neg.keep_events(neg)
    sg_dst_neg.keep_events(neg)
    return sg_src, sg_dst, sg_src_neg, sg_dst_neg


def evaluate_file(file_name, folder, model, sampler):
    src, dst, timestamp, explanations, sg_src, sg_dst, complete_logit, complete_predict = load_explanation(
        file_name, folder, model, sampler)
    sg_src_pos, sg_dst_pos, sg_src_neg, sg_dst_neg = prepare_subgraphs(
        sg_src, sg_dst, explanations)
    values = np.full(len(SPARSITY_THRESHOLDS), timestamp)
    src_values = np.full(len(SPARSITY_THRESHOLDS), src)
    dst_values = np.full(len(SPARSITY_THRESHOLDS), dst)
    logits_pos, predicts_pos, _, _ = get_prediction(
        model, sampler, src_values, dst_values, values, sg_src_pos, sg_dst_pos)
    logits_neg, predicts_neg, _, _ = get_prediction(
        model, sampler, src_values, dst_values, values, sg_src_neg, sg_dst_neg)
    complete_logit, complete_predict = complete_logit[0], complete_predict[0]
    ground_truth = 1.0
    return pd.DataFrame({
        "src": src, "dst": dst, "timestamp": timestamp,
        "ground_truth": ground_truth,
        "logit_complete": complete_logit,
        "predict_complete": complete_predict,
        "logits_pos": logits_pos, 
        "predicts_pos": predicts_pos,
        "logits_neg": logits_neg, 
        "predicts_neg": predicts_neg,
        "fidelity_logit": logits_pos - complete_logit,
        "fidelity_minus_logit": -np.abs(logits_pos - complete_logit),
        "fidelity_plus_logit": np.abs(logits_neg - complete_logit),
        "same_label_minus": (logits_pos > 0) & (complete_logit > 0),
        "same_label_plus": (logits_neg > 0) & (complete_logit > 0),
        "sparsity_thresholds": SPARSITY_THRESHOLDS,
    })


def evaluate_explanations(name: str, model, sampler) -> None:
    directory = os.path.join("Results", "Explanations", CONFIG.data.dataset_name,
                             EXPLAINER_DIRECTORIES[name])
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Explanation directory not found: {directory}")
    files = sorted(file for file in os.listdir(directory) if file.endswith(".npz"))
    if not files:
        raise FileNotFoundError(f"No explanation files found in {directory}")
    frames = [evaluate_file(file, directory, model, sampler)
              for file in tqdm(files, desc="Evaluating explanation files", unit="file")]
    output = os.path.join("Results", "Evaluation", CONFIG.data.dataset_name,
                          f"{EXPLAINER_DIRECTORIES[name]}_explanation_evaluation.csv")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output, index=False)
    print(f"Saved explanation evaluation results to {output}")


def main() -> None:
    
    name = normalize_explainer_name(args.explainer)
    
    model, full_data, train_data, _val_data, _test_data, full_sampler, train_sampler, full_random_sampler = load_model_and_data()
    edge_features = full_sampler.edge_features.detach().cpu().numpy()

    if args.action in ("create", "both"):
        create_explanations(name, model, full_sampler, train_sampler, full_data,
                          train_data, full_random_sampler, edge_features)
    if args.action in ("evaluate", "both"):
        if name == "random_feature":
            raise ValueError("Random feature explainer is only evaluated with the feature explainer.")
        if name == "shapley_feature":
            evaluate_feature_explanations(model, full_sampler, full_data, edge_features, SPARSITY_THRESHOLDS)
        else:
            evaluate_explanations(name, model, full_sampler)


if __name__ == "__main__":
    main()
