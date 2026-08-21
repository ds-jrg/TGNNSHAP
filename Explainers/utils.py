"""
Explanation Utilities for Temporal Graph Neural Network (TGNN) Models
======================================================================

This module defines:
- An abstract base class `Explainer` for building model explainers.
- `ExplanationResult` for evaluating explanation fidelity and sparsity trade-offs.
- Utility functions for computing default feature/timing imputations,
  building subgraphs for evaluation, adjusting timestamp imputations,
  and calculating evaluation metrics such as AUC.

Typical workflow:
1. A subclass of `Explainer` implements `explain_instance()` (producing explanations)
   and `build_coalitions()` (converting explanations to coalitions).
2. `evaluate()` runs across samples, removing events/features by importance,
   and measures resulting prediction fidelity at various sparsity levels.
3. Utility functions help with building subgraphs and creating default feature
   values for masked events.
"""

from abc import ABC, abstractmethod
import numpy as np
import copy
from typing import List, Tuple, Any, Optional
import scipy.special as sc
import pandas as pd
from tqdm import tqdm
from DyGLib.utils.DataLoader import Data
import torch
from DyGLib.utils.utils import BatchSubgraphs, NeighborSampler
from DyGLib.models.modules import TGNN
import traceback
import time
from Config.config import CONFIG
import os

CONFIG = CONFIG()


class Explainer(ABC):
    """
    Abstract base class for TGNN explainers.

    Explainers generate explanations (at event or feature level) for
    a model's predictions, and can be evaluated in terms of sparsity vs fidelity.

    Attributes
    ----------
    model : TGNN
        The temporal GNN model to be explained.
    neighbor_finder : NeighborSampler
        Utility for retrieving temporal neighborhoods.
    data : Data
        Dynamic graph dataset wrapper.
    is_feature_level : bool
        Flag indicating explanation granularity (events vs. features).
    """

    def __init__(self, model: TGNN, neighbor_finder: NeighborSampler, data: Data):
        self.model = model
        self.neighbor_finder = neighbor_finder
        self.data = data
        self.is_feature_level = False

    def explain(self, src: np.ndarray, dst: np.ndarray, timestamp: np.ndarray) -> List[np.ndarray]:
        """
        Generate explanations for multiple (src, dst, timestamp) triplets.

        Parameters
        ----------
        src : np.ndarray
            Source node IDs.
        dst : np.ndarray
            Destination node IDs.
        timestamp : np.ndarray
            Interaction timestamps.

        Returns
        -------
        list of np.ndarray
            List of explanations (format depends on explainer subclass).
        """
        assert src.shape == dst.shape == timestamp.shape, \
            "Src, dst and timestamp need to have the same shape"
        result = []
        for i in range(src.shape[0]):
            result.append(self.explain_instance(src[i], dst[i], timestamp[i]))
        return result

    def evaluate(
        self, src: np.ndarray, dst: np.ndarray, timestamp: np.ndarray,
        ground_truth: np.ndarray, event_features: np.ndarray,
        label_for_prediction: Optional[str] = None, store_coalitions: bool = False,
        intermediate_results_path: str = "",
        timings_path: str = "",
        explainer_name: str = "",
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Evaluate explanation fidelity and sparsity effects for multiple instances.

        Parameters
        ----------
        src, dst, timestamp : np.ndarray
            Source node IDs, destination node IDs, and interaction timestamps.
        ground_truth : np.ndarray
            Ground truth labels for each instance.
        event_features : np.ndarray
            Edge feature matrix for the dataset.
        label_for_prediction : Optional[str]
            Target label for prediction focus (optional).
        intermediate_results_path : str
            Optional CSV path for appending per-instance, non-aggregated
            evaluation results. An empty path disables intermediate storage.

        Returns
        -------
        result_df : pd.DataFrame
            DataFrame with evaluation metrics for both 'Mean' and 'Zero' imputations.
        timings : np.ndarray
            Array of computation times (nanoseconds) per instance.
        """
        assert src.shape == dst.shape == timestamp.shape == ground_truth.shape, \
            "Src, dst, timestamp and ground_truth need to have the same shape."
        mean_values, mean_delta_timings = compute_default_values(self.data, event_features, label_for_prediction=label_for_prediction)

        timings = []
        result_mean = []
        result_zero = []

        intermediate_columns = [
            "Original prediction", "Ground truth", "y",
            "Sparsity thresholds", "Fidelity to prediction",
            "Fidelity to prediction (logit)",
            "Deviation to ground truth", "Accuracy",
            "Source", "Destination", "Timestamp", "Instance index",
            "Remove technique",
        ]
        if intermediate_results_path:
            parent = os.path.dirname(intermediate_results_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            if not os.path.exists(intermediate_results_path):
                pd.DataFrame(columns=intermediate_columns).to_csv(
                    intermediate_results_path, index=False
                )

        timing_columns = ["Time(ns)", "Time(s)", "Explainer", "Stage", "Instance Index"]
        if timings_path:
            parent = os.path.dirname(timings_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            if not os.path.exists(timings_path):
                pd.DataFrame(columns=timing_columns).to_csv(timings_path, index=False)

        def append_timing(elapsed_ns, stage, instance_index):
            if not timings_path:
                return
            pd.DataFrame([{
                "Time(ns)": elapsed_ns,
                "Time(s)": elapsed_ns / 1_000_000_000,
                "Explainer": explainer_name,
                "Stage": stage,
                "Instance Index": instance_index,
            }]).to_csv(timings_path, mode="a", header=False, index=False)

        def append_intermediate_results(result, remove_technique, instance_index):
            if not intermediate_results_path:
                return
            frame = pd.DataFrame(result, columns=intermediate_columns[:8])
            frame["Source"] = src[instance_index]
            frame["Destination"] = dst[instance_index]
            frame["Timestamp"] = timestamp[instance_index]
            frame["Instance index"] = instance_index
            frame["Remove technique"] = remove_technique
            frame.to_csv(
                intermediate_results_path,
                mode="a",
                header=False,
                index=False,
            )

        for i in tqdm(range(src.shape[0])):
            try:
                start = time.time_ns()
                explanation = self.explain_instance(src[i], dst[i], timestamp[i], silent=True)
                end = time.time_ns()
                timings.append(end - start)
                append_timing(end - start, "Explain", i)

                coalitions, sg_src, sg_dst = self.build_coalitions(explanation)
                if(store_coalitions):
                    file_path = f"Logs/Coalitions/{CONFIG.data.dataset_name}/{self.__class__.__name__}/"
                    os.makedirs(file_path, exist_ok=True)
                    np.savez_compressed(f"{file_path}{src[i]}_to_{dst[i]}_{timestamp[i]}.npz", coalitions=coalitions)    
                _, _, _, imputation_data = default_values_subgraph(
                    src[i], dst[i], timestamp[i], self.neighbor_finder, self.data,
                    mean_delta_timings, mean_values, sg_src=sg_src, sg_dst=sg_dst
                )
                evaluator = ExplanationResult(self.model, self.neighbor_finder)

                # Evaluate with mean imputation
                if self.is_feature_level:
                    evaluation, _ = evaluator.evaluate_on_feature_level(
                        src[i], dst[i], timestamp[i], ground_truth[i],
                        coalitions, imputation_data, sg_src=sg_src, sg_dst=sg_dst
                    )
                else:
                    evaluation, _ = evaluator.evaluate(
                        src[i], dst[i], timestamp[i], ground_truth[i],
                        coalitions, imputation_data, sg_src=sg_src, sg_dst=sg_dst
                    )
                result_mean.append(evaluation)
                append_intermediate_results(evaluation, "Mean", i)

                # Evaluate with zero imputation
                if self.is_feature_level:
                    evaluation, _ = evaluator.evaluate_on_feature_level(
                        src[i], dst[i], timestamp[i], ground_truth[i],
                        coalitions, None, sg_src=sg_src, sg_dst=sg_dst
                    )
                else:
                    evaluation, _ = evaluator.evaluate(
                        src[i], dst[i], timestamp[i], ground_truth[i],
                        coalitions, None, sg_src=sg_src, sg_dst=sg_dst
                    )
                result_zero.append(evaluation)
                append_intermediate_results(evaluation, "Zero", i)

                torch.cuda.empty_cache()
            except Exception:
                print(f"Exception for {src[i]} to {dst[i]} at {timestamp[i]} occured: {traceback.format_exc()}")

        # Stack per-instance results and compute averages + GEF metric
        result_mean = np.stack(result_mean, axis=2)
        result_zero = np.stack(result_zero, axis=2)
        result_versions = [result_mean, result_zero]

        if CONFIG.model.task == "classification":
            for i in range(len(result_versions)):
                result = result_versions[i]
                y_expl = result[:, 2, :]
                y_ori = result[:, 0, :]
                kl = sc.kl_div(y_ori, y_expl).sum(axis=1)
                gef = np.exp(-kl)
                result_versions[i] = np.concat(
                    [result.mean(axis=2), gef.reshape((-1, 1))], axis=1
                )
        else:
            for i in range(len(result_versions)):
                result_versions[i] = np.concat(
                    [result_versions[i].mean(axis=2),
                     np.zeros((result_versions[i].shape[0], 1))], axis=1
                )

        # Convert to DataFrames with labels
        result_mean_pd = pd.DataFrame(
            result_versions[0],
            columns=[
                "Original prediction", "Ground truth", "y",
                "Sparsity thresholds", "Fidelity to prediction",
                "Fidelity to prediction (logit)",
                "Deviation to ground truth", "GEF", "Accuracy"
            ]
        )
        result_mean_pd["Remove technique"] = "Mean"

        result_zero_pd = pd.DataFrame(
            result_versions[1],
            columns=[
                "Original prediction", "Ground truth", "y",
                "Sparsity thresholds", "Fidelity to prediction",
                "Fidelity to prediction (logit)",
                "Deviation to ground truth", "GEF", "Accuracy"
            ]
        )
        result_zero_pd["Remove technique"] = "Zero"

        timings = np.array(timings).reshape((-1, 1))
        return pd.concat([result_zero_pd, result_mean_pd]), timings

    @abstractmethod
    def explain_instance(self, src: int, dst: int, timestamp: int, silent: bool = False) -> Any:
        """
        Explain a single (src, dst, timestamp) instance.

        Parameters
        ----------
        src, dst : int
            Source and destination node IDs.
        timestamp : int
            Interaction time.
        silent : bool
            Whether to suppress verbose outputs.

        Returns
        -------
        Any
            Implementation-dependent explanation object (events, shap values, etc.).
        """
        pass
    
    @abstractmethod
    def initialize(self, *args, **kwargs):
        """
        Optional initialization for the explainer, e.g., model training.
        """
        pass

    @abstractmethod
    def build_coalitions(
        self, explanation: Any
    ) -> Tuple[Any, Optional[BatchSubgraphs], Optional[BatchSubgraphs]]:
        """
        Convert explanation output into coalition sets for removal testing.

        Returns
        -------
        coalitions : Any
            Matrix or list representation of feature/event inclusion order.
        sg_src, sg_dst : Optional[BatchSubgraphs]
            Cached source and destination subgraphs to reuse in evaluation.
        """
        pass

class ExplanationResult:
    """
    Evaluation helper for explanations.

    This class measures prediction fidelity as events or features
    are progressively removed according to the explanation's importance order.

    Attributes
    ----------
    model : TGNN
        Temporal GNN model to run predictions.
    neighbor_finder : NeighborSampler
        For retrieving temporal neighbors and subgraphs.
    """

    def __init__(self, model: TGNN, neighbor_finder: NeighborSampler):
        self.model = model
        self.neighbor_finder = neighbor_finder

    def evaluate(
        self, src: int, dst: int, timestamp: int, ground_truth: float,
        coalitions: np.ndarray, default_values: Optional[dict] = None,
        sg_src: Optional[BatchSubgraphs] = None, sg_dst: Optional[BatchSubgraphs] = None
    ):
        """
        Evaluate event-level importance by progressively removing events according to coalitions.

        Parameters
        ----------
        src, dst : int
            Source and destination node IDs.
        timestamp : int
            Interaction timestamp.
        ground_truth : float
            Ground truth label for link prediction.
        coalitions : np.ndarray
            Coalition matrix defining which events are included at each step.
        default_values : Optional[dict]
            Imputation values for missing events (mean values).
        sg_src, sg_dst : Optional[BatchSubgraphs]
            Optional cached subgraphs to speed up evaluation.

        Returns
        -------
        result : np.ndarray
            Evaluation metrics for each sparsity threshold.
        coalitions_per_sparsity : np.ndarray
            Coalition sets picked at each sparsity level.
        """
        sparsity_thresholds = np.linspace(0, 1, 50)
        sparsity_thresholds = 0.5 * (1 + np.tanh(7 * (sparsity_thresholds - 0.5)))
        sparsity_thresholds = np.concat([[0.0], sparsity_thresholds, [1.0]])
        subgraph_src, subgraph_dst, sg_src, sg_dst, original_prediction = self._init_eval(src, dst, timestamp, sg_src, sg_dst)

        events = np.unique(np.concat([subgraph_dst.get_events(), subgraph_src.get_events()], axis=1))
        events = events[events != 0]
        num_events = events.shape[0]
        subgraph_src.repeat_nodes(sparsity_thresholds.shape[0])
        subgraph_dst.repeat_nodes(sparsity_thresholds.shape[0])

        coalitions_per_sparsity = np.zeros((sparsity_thresholds.shape[0], coalitions.shape[1] + 1), dtype=int)
        coalition_sparsities = ((coalitions != 0) & np.isin(coalitions, events)).sum(axis=1) / num_events
        for i, s in enumerate(sparsity_thresholds):
            mask = coalition_sparsities <= s
            if mask.any():
                coalitions_per_sparsity[i, :-1] = coalitions[mask][coalition_sparsities[mask].argmax()]

        subgraph_src.keep_events(coalitions_per_sparsity, default_values)
        subgraph_dst.keep_events(coalitions_per_sparsity, default_values)
        result = self._run_eval(src, dst, timestamp, ground_truth, subgraph_src, subgraph_dst, sg_src, sg_dst, sparsity_thresholds, original_prediction)
        return result, coalitions_per_sparsity

    def evaluate_on_feature_level(
        self, src: int, dst: int, timestamp: int, ground_truth: float,
        importance_order: np.ndarray, default_values: Optional[dict] = None,
        sg_src: Optional[BatchSubgraphs] = None, sg_dst: Optional[BatchSubgraphs] = None
    ):
        """
        Evaluate feature-level importance by progressively masking features for an event.

        Parameters
        ----------
        src, dst : int
            Source and destination node IDs.
        timestamp : int
            Interaction timestamp.
        ground_truth : float
            Ground truth label.
        importance_order : np.ndarray
            Ordered list of (event_id, feature_id) tuples by importance.
        default_values : Optional[dict]
            Imputation defaults for missing values.
        sg_src, sg_dst : Optional[BatchSubgraphs]
            Cached subgraphs.

        Returns
        -------
        result : np.ndarray
            Evaluation metrics for each sparsity threshold.
        coalitions_per_sparsity : list
            List of coalition sets picked at each sparsity level.
        """
        sparsity_thresholds = np.linspace(0, 1, 50)
        sparsity_thresholds = 0.5 * (1 + np.tanh(6 * (sparsity_thresholds - 0.5)))
        sparsity_thresholds = np.concat([[0.0], sparsity_thresholds, [1.0]])
        subgraph_src, subgraph_dst, sg_src, sg_dst, original_prediction = self._init_eval(src, dst, timestamp, sg_src, sg_dst)

        subgraph_src.repeat_nodes(sparsity_thresholds.shape[0])
        subgraph_dst.repeat_nodes(sparsity_thresholds.shape[0])

        coalitions_per_sparsity = []
        for i, s in enumerate(sparsity_thresholds):
            j = int(len(importance_order) * s)
            feats = np.concat((importance_order[:j], np.array([(0, x) for x in range(subgraph_src.get_num_features() + 2)], dtype=int)))
            coalitions_per_sparsity.append(feats)

        subgraph_src.keep_features(coalitions_per_sparsity, default_values)
        subgraph_dst.keep_features(coalitions_per_sparsity, default_values)
        
        events_per_sparsity = [np.unique(c[:,0]) for c in coalitions_per_sparsity]   
        #Pad array to homogeneous shape
        max_len = max([len(e) for e in events_per_sparsity])
        for i in range(len(events_per_sparsity)):
            events_per_sparsity[i] = np.concat([events_per_sparsity[i], np.zeros((max_len - len(events_per_sparsity[i]),), dtype=int)])
        events_per_sparsity = np.array(events_per_sparsity)
        subgraph_src.keep_events(events_per_sparsity, default_values)
        subgraph_dst.keep_events(events_per_sparsity, default_values)
        
        result = self._run_eval(src, dst, timestamp, ground_truth, subgraph_src, subgraph_dst, sg_src, sg_dst, sparsity_thresholds, original_prediction)
        return result, coalitions_per_sparsity

    def _init_eval(self, src: int, dst: int, timestamp: int, sg_src: Optional[BatchSubgraphs] = None, sg_dst: Optional[BatchSubgraphs] = None):
        """
        Prepare subgraphs and obtain the original model prediction for the instance.
        """
        if sg_src is None:
            subgraph_src = self.neighbor_finder.get_multi_hop_neighbors(CONFIG.model.num_layers, np.array([src]), np.array([timestamp]), num_neighbors=CONFIG.model.num_neighbors)
            event_feat_src = self.neighbor_finder.get_edge_features_for_multi_hop(subgraph_src[1])
            subgraph_src = BatchSubgraphs(*subgraph_src, event_feat_src)
            sg_src = copy.deepcopy(subgraph_src)
        else:
            subgraph_src = copy.deepcopy(sg_src)

        if sg_dst is None:
            subgraph_dst = self.neighbor_finder.get_multi_hop_neighbors(CONFIG.model.num_layers, np.array([dst]), np.array([timestamp]), num_neighbors=CONFIG.model.num_neighbors)
            event_feat_dst = self.neighbor_finder.get_edge_features_for_multi_hop(subgraph_dst[1])
            subgraph_dst = BatchSubgraphs(*subgraph_dst, event_feat_dst)
            sg_dst = copy.deepcopy(subgraph_dst)
        else:
            subgraph_dst = copy.deepcopy(sg_dst)

        subgraph_src.to(CONFIG.model.device)
        subgraph_dst.to(CONFIG.model.device)
        sg_src.to(CONFIG.model.device)
        sg_dst.to(CONFIG.model.device)

        original_prediction = self.model(
            src_node_ids=np.array([src]),
            dst_node_ids=np.array([dst]),
            node_interact_times=np.array([timestamp]),
            src_subgraphs=subgraph_src,
            dst_subgraphs=subgraph_dst,
            num_neighbors=CONFIG.model.num_neighbors,
            time_gap=CONFIG.model.time_gap,
            edges_are_positive=False
        )
        return subgraph_src, subgraph_dst, sg_src, sg_dst, original_prediction

    def _run_eval(
        self, src: int, dst: int, timestamp: int, ground_truth: float,
        subgraph_src: BatchSubgraphs, subgraph_dst: BatchSubgraphs,
        sg_src: BatchSubgraphs, sg_dst: BatchSubgraphs,
        sparsity_thresholds: np.ndarray, original_prediction: np.ndarray
    ):
        """
        Run model predictions for each sparsity level and compute evaluation metrics.
        """
        assert sg_src == subgraph_src[-1:], "The last coalition does not contain all events!"
        assert sg_dst == subgraph_dst[-1:], "The last coalition does not contain all events!"
        srcs = np.full((sparsity_thresholds.shape[0],), src)
        dsts = np.full((sparsity_thresholds.shape[0],), dst)
        tss = np.full((sparsity_thresholds.shape[0],), timestamp)

        y = self.model(
            src_node_ids=srcs,
            dst_node_ids=dsts,
            node_interact_times=tss,
            src_subgraphs=subgraph_src,
            dst_subgraphs=subgraph_dst,
            num_neighbors=CONFIG.model.num_neighbors,
            time_gap=CONFIG.model.time_gap,
            edges_are_positive=False
        )

        original_prediction = original_prediction.cpu().detach()
        y = y.cpu().detach()
        fidelity2 = np.zeros_like(y)

        if CONFIG.model.task != "regression":
            fidelity2[ground_truth == 1] = y - original_prediction
            fidelity2[ground_truth == 0] = original_prediction - y
            y = y.sigmoid()
            original_prediction = original_prediction.sigmoid()

        original_prediction = original_prediction.numpy()
        y = y.numpy()

        fidelity = -np.abs(original_prediction - y)
        distance_to_ground_truth = -np.abs(ground_truth - y)
        
        accuracy = ((y >= 0.5) == (original_prediction >= 0.5)).astype(float)

        result = np.zeros((sparsity_thresholds.shape[0], 8))
        result[:, 0] = original_prediction.reshape((-1,))
        result[:, 1] = ground_truth
        result[:, 2] = y.reshape((-1,))
        result[:, 3] = sparsity_thresholds.reshape((-1,))
        result[:, 4] = fidelity.reshape((-1,))
        result[:, 5] = fidelity2.reshape((-1,))
        result[:, 6] = distance_to_ground_truth.reshape((-1,))
        result[:, 7] = accuracy.reshape((-1,))

        return result


def compute_default_values(data: Data, event_features: np.ndarray, max_timing: Optional[float] = None, label_for_prediction: Optional[str] = None):
    """
    Compute mean feature vectors and mean timing offsets (delta_t) per event type.

    These statistics are used for imputing missing events during explanation evaluation.

    Parameters
    ----------
    data : Data
        Dynamic graph dataset.
    event_features : np.ndarray
        Edge-level features for the graph.
    max_timing : Optional[float]
        If specified, only compute over events with timestamps <= max_timing.
    label_for_prediction : Optional[str]
        If specified, only compute over events with this label.

    Returns
    -------
    mean_values : dict
        Mapping from event type to mean feature vector.
    mean_delta_timings : dict
        Mapping from event type to average time gap to previous event of same type.
    """
    mean_delta_timings = {}
    mean_values = {}
    if data.dataset is not None:
        d = data.dataset[["i", "ts", "type"]].copy()
        if label_for_prediction is not None:
            d_base = d[["i", "ts"]][d.type == label_for_prediction].sort_values(by="ts")
        else:
            d_base = d[["i", "ts"]].sort_values(by="ts")
            
        if max_timing is not None:
            d = d[d.ts <= max_timing]
            d_base = d_base[d_base.ts <= max_timing]

        for l, _ in d.type.value_counts().items():
            d_l = d[["i", "ts"]][d.type == l].sort_values(by="ts").rename(columns={'ts': f'ts_{l}'})
            d_base = pd.merge_asof(d_base, d_l, by="i", left_on="ts", right_on=f'ts_{l}', allow_exact_matches=False)
            d_base[l] = d_base.ts - d_base.loc[:, f'ts_{l}']
            if data.edge_ids[0] != 0:
                mean_values[l] = np.mean(event_features[1:][data.types == l], axis=0)
            else:
                mean_values[l] = np.mean(event_features[data.types == l], axis=0)

        mean_delta_timings = d_base[d.type.value_counts().index].mean().to_dict()
    return mean_values, mean_delta_timings


def default_values_subgraph(
    src: int, dst: int, timestamp: int, neighbor_finder: NeighborSampler,
    data: Data, mean_delta_timings: dict, mean_values: dict,
    sg_src: Optional[BatchSubgraphs] = None, sg_dst: Optional[BatchSubgraphs] = None
):
    """
    Construct source/destination subgraphs and prepare default imputation values for events.

    This helper ensures masked events are replaced with mean values/timings.

    Returns
    -------
    subgraphs_src, subgraphs_dst : BatchSubgraphs
        Computed/buffered source and destination subgraphs.
    event_ids : np.ndarray
        List of event IDs present in either subgraph.
    imputation_data : dict
        Mapping from event ID to default feature vector tensor.
    """
    if sg_src is None:
        subgraphs_src = neighbor_finder.get_multi_hop_neighbors(CONFIG.model.num_layers, np.array([src]), np.array([timestamp]), num_neighbors=CONFIG.model.num_neighbors)
        event_feat_src = neighbor_finder.get_edge_features_for_multi_hop(subgraphs_src[1])
        subgraphs_src = BatchSubgraphs(*subgraphs_src, event_feat_src)
    else:
        subgraphs_src = copy.deepcopy(sg_src)

    if sg_dst is None:
        subgraphs_dst = neighbor_finder.get_multi_hop_neighbors(CONFIG.model.num_layers, np.array([dst]), np.array([timestamp]), num_neighbors=CONFIG.model.num_neighbors)
        event_feat_dst = neighbor_finder.get_edge_features_for_multi_hop(subgraphs_dst[1])
        subgraphs_dst = BatchSubgraphs(*subgraphs_dst, event_feat_dst)
    else:
        subgraphs_dst = copy.deepcopy(sg_dst)

    subgraphs_src.to(CONFIG.model.device)
    subgraphs_dst.to(CONFIG.model.device)
    event_ids = np.concat([subgraphs_src.get_events(), subgraphs_dst.get_events()]).reshape((-1,))
    event_ids = np.unique(event_ids[event_ids != 0])
    imputation_data = {}
    for id, l, ts in zip(event_ids, data.types[event_ids], data.node_interact_times[event_ids]):
        imputation_data[id] = torch.from_numpy(np.concat(([0, 0], mean_values[l]))).float().to(CONFIG.model.device)

    calc_mean_timing(timestamp, subgraphs_src, imputation_data, mean_delta_timings, data)
    calc_mean_timing(timestamp, subgraphs_dst, imputation_data, mean_delta_timings, data)
    return subgraphs_src, subgraphs_dst, event_ids, imputation_data


def calc_mean_timing(timestamp: int, subgraph: BatchSubgraphs, imputation_data: dict, mean_delta_timings: dict, data: Data):
    """
    Adjust imputed timestamps for events by subtracting the mean delta time for their type.
    """
    ts = np.array([timestamp])
    for e_ids, timestamps in zip(subgraph.events, subgraph.timestamps):
        is_no_event_mask = e_ids == 0
        types = data.types[e_ids]
        ts = ts.repeat(CONFIG.model.num_neighbors) # TODO Replace with num neighbors
        new_timestamps = np.array([t - mean_delta_timings[l] if l is not np.nan else 0 for l, t in np.concat([types.reshape((-1, 1)), ts.reshape((-1, 1))], axis=1)])
        new_timestamps = new_timestamps.reshape((1, -1))
        new_timestamps[is_no_event_mask] = 0
        ts = timestamps.copy()
        for id, new_ts in np.concat((e_ids.reshape((-1, 1)), new_timestamps.reshape((-1, 1))), axis=1):
            if id != 0:
                imputation_data[id][1] = new_ts