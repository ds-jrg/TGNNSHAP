"""
Shapley Value-based Explainers for Temporal Graph Neural Networks (TGNN).

This module contains two explainers:
- ShapleyExplainerEvents: Explains predictions by attributing importance to events.
- ShapleyExplainerEventsPositive: Event-level variant using positive attributions only.
- ShapleyExplainerFeatures: Explains predictions by attributing importance to event features.

Both use variants of Shapley Value computation (KernelSHAP, Monte Carlo, Permutation).
"""

from Explainers.utils import Explainer, ExplanationResult, compute_default_values, default_values_subgraph
from DyGLib.utils.DataLoader import Data
from DyGLib.utils.utils import NeighborSampler, BatchSubgraphs, concat_subgraphs
from DyGLib.models.modules import TGNN
from Config.config import CONFIG
import numpy as np
import torch
from typing import Optional, Union
import copy
import shap
from tqdm import tqdm
import time

CONFIG = CONFIG()

import subprocess as sp
import os

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def _predict_model_batchwise(
    model: TGNN,
    srcs: np.ndarray,
    dsts: np.ndarray,
    time_stamps: np.ndarray,
    src_subgraphs: BatchSubgraphs,
    dst_subgraphs: BatchSubgraphs,
) -> torch.Tensor:
    """Query the model in configurable chunks and preserve input order."""
    batch_size = CONFIG.shapley.batch_size
    if batch_size <= 0:
        raise ValueError("Shapley.batch_size must be positive")

    predictions = []
    for start in range(0, len(srcs), batch_size):
        end = min(start + batch_size, len(srcs))
        predictions.append(model(
            srcs[start:end], dsts[start:end], time_stamps[start:end],
            src_subgraphs=src_subgraphs[start:end],
            dst_subgraphs=dst_subgraphs[start:end],
            time_gap=CONFIG.model.time_gap,
            edges_are_positive=False,
            num_neighbors=CONFIG.model.num_neighbors
        ).cpu().detach())
        torch.cuda.empty_cache()
    return torch.cat(predictions, dim=0)


class ShapleyExplainerEvents(Explainer):
    """
    Explains TGNN predictions by computing Shapley values over events
    in the local temporal subgraph surrounding a target interaction.

    Attributes
    ----------
    model : TGNN
        The temporal graph neural network model being explained.
    neighbor_finder : NeighborSampler
        Allows retrieval of temporal neighborhood for a given node and time.
    data : Data
        Graph dataset wrapper provided by DyGLib.
    event_features : np.ndarray
        Matrix of event-level features.
    """

    def __init__(self, model: TGNN, neighbor_finder: NeighborSampler, data: Data, 
                 event_features: np.ndarray, node_names: Optional[np.ndarray] = None, 
                 algorithm: str = "KernelSHAP", recompute_mean_on_prevoius_events: bool = False,
                 label_for_prediction: Optional[str] = None):
        super().__init__(model, neighbor_finder, data)
        self.event_features = event_features
        self.node_names = node_names
        self.algorithm = algorithm
        self.recompute_mean_on_prevoius_events = recompute_mean_on_prevoius_events
        self.label_for_prediction = label_for_prediction

    def initialize(self):
        """
        Precomputes default event feature and timing values 
        used for masked events.
        """
        if (not self.recompute_mean_on_prevoius_events):
            self.mean_values, self.mean_delta_timings = compute_default_values(
                self.data, self.event_features, label_for_prediction=self.label_for_prediction
            )

    def _explain_instance(self, src: int, dst: int, timestamp: float, subgraphs_src: BatchSubgraphs, subgraphs_dst: BatchSubgraphs, event_ids: np.ndarray, imputation_data: dict, silent: bool = False):
        """
        Internal method to compute Shapley values for a given set of subgraphs and events.

        Parameters
        ----------
        src : int
            Source node ID.
        dst : int
            Destination node ID.
        timestamp : float
            Timestamp of the interaction.
        subgraphs_src : BatchSubgraphs
            Source-side local subgraph.
        subgraphs_dst : BatchSubgraphs
            Destination-side local subgraph.
        event_ids : np.ndarray
            Array of event IDs in the subgraph.
        imputation_data : dict
            Mapping from event ID to default feature values for masking.

        Returns
        -------
        shap_values : shap.Explanation
            Shapley values for each event in the subgraph.
        """
        assert len(event_ids) != 0, "The computational subgraph contains no events!"
        
        def val(x):
            """
            Prediction function for SHAP that masks certain events.
            """
            srcs = np.full((len(x),), src)
            dsts = np.full((len(x),), dst)
            time_stamps = np.full((len(x),), timestamp)

            sg_src = copy.deepcopy(subgraphs_src)
            sg_dst = copy.deepcopy(subgraphs_dst)
            sg_src.repeat_nodes(len(x))
            sg_dst.repeat_nodes(len(x))

            # Mask events where x==0
            to_mask_events = x == 0
            sg_src.mask_events(event_ids, event_mask=to_mask_events, data_per_event=imputation_data)
            sg_dst.mask_events(event_ids, event_mask=to_mask_events, data_per_event=imputation_data)

            # Model prediction
            y = _predict_model_batchwise(
                self.model, srcs, dsts, time_stamps, sg_src, sg_dst
            )
            if CONFIG.model.task == "classification":
                y = y.sigmoid()

            return y.cpu().detach().numpy().reshape((-1,))

        # Human-readable feature names for events
        if self.node_names is not None:
            labels = [f"{self.node_names[self.data.src_node_ids[e]]} to {self.node_names[self.data.dst_node_ids[e]]} @ {self.data.node_interact_times[e]}" for e in event_ids]
        else:
            labels = [f"{self.data.src_node_ids[e]} to {self.data.dst_node_ids[e]} @ {self.data.node_interact_times[e]}" for e in event_ids]

        # Instantiate SHAP on binary event-inclusion mask
        if self.algorithm == "KernelSHAP":
            explainer = shap.explainers.KernelExplainer(
                val, data=np.zeros((1, len(event_ids))), feature_names=labels
            )
            shap_values = explainer(np.array([event_ids]).reshape(1, -1), silent=silent,  l1_reg=False) # type: ignore
        elif self.algorithm == "Exact":
            explainer = shap.explainers.ExactExplainer(val, masker=np.zeros((1, len(event_ids))), feature_names=labels)
            shap_values = explainer(np.array([event_ids]).reshape(1, -1), silent=silent)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        return event_ids, shap_values
    
    def explain_instance(self, src, dst, timestamp, silent=False):
        """
        Compute event-level Shapley values for a given node pair at a given time.

        Parameters
        ----------
        src : int
            Source node ID.
        dst : int
            Destination node ID.
        timestamp : int
            Time of the interaction to explain.
        silent : bool
            If True, suppress progress output.

        Returns
        -------
        tuple
            event_ids : ndarray
                Array of event indices in the subgraph.
            shap_values : shap.Explanation
                Shapley values for each event in the subgraph.
        """
        if self.recompute_mean_on_prevoius_events:
            self.mean_values, self.mean_delta_timings = compute_default_values(
                self.data, self.event_features, label_for_prediction=self.label_for_prediction, max_timing=timestamp
            )
        
        # Get local temporal subgraphs and imputation defaults
        subgraphs_src, subgraphs_dst, event_ids, imputation_data = default_values_subgraph(
            src, dst, timestamp, self.neighbor_finder, self.data,
            self.mean_delta_timings, self.mean_values
        )
        return self._explain_instance(src, dst, timestamp, subgraphs_src, subgraphs_dst, event_ids, imputation_data, silent)

    def build_coalitions(self, explanation):
        """
        Order events by absolute Shapley value and construct coalition sets.

        Parameters
        ----------
        explanation : tuple
            Output of `explain_instance`.

        Returns
        -------
        coalitions : ndarray
            Row i contains the top-(i+1) events in ranked order.
        """
        event_ids, shap_values = explanation
        events = event_ids.reshape((-1,))
        shap_values = shap_values.values.reshape((-1,))
        
        shuffle_mask = torch.randperm(len(events))
        events = events[shuffle_mask]
        shap_values = shap_values[shuffle_mask]
        
        sorting = (-np.abs(shap_values)).argsort()
        events = events[sorting]

        coalitions = np.zeros((len(events), len(events)))
        for i in range(len(events)):
            coalitions[i, :i + 1] = events[:i + 1]

        return coalitions, None, None


class ShapleyExplainerEventsPositive(ShapleyExplainerEvents):
    """Event-level Shapley explainer using only positive attributions.

    Positive events are ordered by their raw Shapley value and added to
    cumulative coalitions.  The final coalition always contains every event
    so that evaluation can reach the complete computational subgraph.
    """

    def build_coalitions(self, explanation):
        """Build cumulative coalitions from positive Shapley values only."""
        event_ids, shap_values = explanation
        events = np.asarray(event_ids).reshape((-1,))
        values = np.asarray(shap_values.values).reshape((-1,))

        positive = values > 0
        positive_events = events[positive]
        positive_values = values[positive]
        sorting = np.argsort(-positive_values, kind="stable")
        positive_events = positive_events[sorting]

        # The extra row is intentionally retained even when all events are
        # positive, in order to always append the complete coalition.
        coalitions = np.zeros(
            (len(positive_events) + 1, len(events)),
            dtype=np.float64,
        )
        for i in range(len(positive_events)):
            coalitions[i, :i + 1] = positive_events[:i + 1]

        coalitions[-1, :len(events)] = events
        return coalitions, None, None


class ShapleyExplainerFeatures(Explainer):
    """
    Explains TGNN predictions by computing Shapley values over 
    individual event features (structure, timing, attributes).

    Supports multiple methods:
    - KernelSHAP (results not reported, as the implementation is complicated and the results do not exceed Monte Carlo approximation)
    - Monte Carlo approximation
    - Permutation-based importance

    Attributes
    ----------
    feature_names : Optional[np.ndarray]
        Labels for event features for interpretability.
    shapley_alg : str
        Algorithm to use: "KernelSHAP", "MonteCarlo", or "Permutation".
    top_k : Optional[int]
        If set, only top-k important events from the event-level 
        explanation are expanded to feature-level analysis.
    """

    def __init__(self, model: TGNN, neighbor_finder: NeighborSampler, data: Data,
                 event_features: np.ndarray, feature_names: Optional[Union[np.ndarray, bool]] = None,
                 shapley_alg="KernelSHAP", top_k: Optional[int] = None, recompute_mean_on_prevoius_events: bool = False, 
                 label_for_prediction: Optional[str] = None,
                 node_names: Optional[np.ndarray] = None):
        super().__init__(model, neighbor_finder, data)
        self.event_features = event_features
        self.feature_names = feature_names
        self.shapley_alg = shapley_alg
        self.is_feature_level = True
        self.node_names = node_names
        self.top_k = top_k
        self.recompute_mean_on_prevoius_events = recompute_mean_on_prevoius_events
        self.label_for_prediction = label_for_prediction

    def initialize(self):
        """Compute default values for imputing masked features."""
        if not self.recompute_mean_on_prevoius_events:
            self.mean_values, self.mean_delta_timings = compute_default_values(
                self.data, self.event_features, label_for_prediction=self.label_for_prediction
            )

    def build_coalitions(self, explanation):
        """
        Construct coalition sets where each player is an (event_id, feature_id) pair.

        Parameters
        ----------
        explanation : tuple
            Output of `explain_instance`.

        Returns
        -------
        ndarray
            Players matrix: rows are (event_id, feature_id).
        """
        explanation_flatten, remaining_ids, _, _ = explanation
        num_players_per_event = self.event_features.shape[1] + 2  # 2 extra for structure & timing
        players = [(x[0], x[5]) for x in explanation_flatten]
        players = np.array(players, dtype=int)

        # Add players from remaining (unexplained) events
        remaining_players = np.zeros((len(remaining_ids) * num_players_per_event, 2), dtype=int)
        for i, e_id in enumerate(remaining_ids):
            remaining_players[i * num_players_per_event: (i+1) * num_players_per_event, :] = \
                np.array([(e_id, i) for i in range(num_players_per_event)], dtype=int)

        players = np.concat((players, remaining_players))
        return players, None, None

    def explain_instance(self, src: int, dst: int, timestamp: int, silent=False,
                         event_id: Optional[int] = None, max_num_samples: Optional[int] = None):
        """
        Explain features contributing to a given node interaction.

        If `event_id` is provided, only that event is explained in detail.
        If `top_k` is set, only top-k events from event-level SHAP are expanded.

        Parameters
        ----------
        src : int
            Source node ID.
        dst : int
            Destination node ID.
        timestamp : int
            Interaction time.
        silent : bool
            Suppress progress bars if True.
        event_id : Optional[int]
            Specific event ID to focus on.
        max_num_samples : int
            Limit for Monte Carlo sampling. If omitted, the value from
            ``Shapley.max_num_samples`` is used.

        Returns
        -------
        tuple
            Flattened feature SHAP values, remaining event IDs, 
            remaining event SHAP scores, and baseline value.
        """
        if self.recompute_mean_on_prevoius_events:
            self.mean_values, self.mean_delta_timings = compute_default_values(
                self.data, self.event_features, label_for_prediction=self.label_for_prediction, max_timing=timestamp
            )
        if max_num_samples is None:
            max_num_samples = CONFIG.shapley.max_num_samples
        # Get local computational subgraph
        subgraphs_src, subgraphs_dst, event_ids, imputation_data = default_values_subgraph(
            src, dst, timestamp, self.neighbor_finder, self.data,
            self.mean_delta_timings, self.mean_values
        )
        assert len(event_ids) != 0, "The computational subgraph contains no events!"

        # First run event-level explainer to get important events
        event_explainer = ShapleyExplainerEvents(
            self.model, self.neighbor_finder, self.data, self.event_features
        )
        event_explainer.mean_delta_timings = self.mean_delta_timings
        event_explainer.mean_values = self.mean_values
        e_ids, shapley_values = event_explainer.explain_instance(
            src, dst, timestamp, silent=True
        )
        baseline = shapley_values.base_values[0]
        torch.cuda.empty_cache()

        # Select events to explain at feature-level
        if event_id is not None:
            ids_to_explain = np.array([event_id])
            remaining_ids = (e_ids[e_ids != event_id]).flatten()
            remaining_shapley_values = (shapley_values.values[0][e_ids != event_id]).flatten()
        elif self.top_k is not None:
            sorting = (np.argsort(-np.abs(shapley_values.values))).flatten()
            ids_to_explain = (e_ids[sorting[:self.top_k]]).flatten()
            remaining_ids = (e_ids[sorting[self.top_k:]]).flatten()
            remaining_shapley_values = (shapley_values.values[0][sorting[self.top_k:]]).flatten()
        else:
            raise ValueError("Either `event_id` or `top_k` must be specified for feature-level explanation.")

        # Prepare events for explanation
        id_to_pos_dict = {e : pos for pos, e in enumerate(self.data.edge_ids)}
        id_pos = np.array([id_to_pos_dict[e] for e in ids_to_explain], dtype=int)
        event_timings = self.data.dataset.ts[id_pos] if self.data.dataset is not None else []
        event_srcs = self.data.src_node_ids[id_pos]
        event_dsts = self.data.dst_node_ids[id_pos]
        event_types = self.neighbor_finder.edge_labels[id_pos]
        events = list(zip(ids_to_explain, event_srcs, event_dsts, event_types, event_timings))
        pbar = tqdm(events, total=len(events), desc='Explain event') if not silent else events

        explanation_flatten = []
        for e_id, s, d, label, timing in pbar:
            if self.shapley_alg == "KernelSHAP":
                expl = self.explain_event_kernel(
                    src, dst, timestamp, e_id, subgraphs_src, subgraphs_dst, imputation_data, silent
                )
                features, features_values, values = expl.feature_names, expl[0].data, expl[0].values
            elif self.shapley_alg == "MonteCarlo":
                expl = self.explain_event_monte_carlo(
                    src, dst, timestamp, e_id, subgraphs_src, subgraphs_dst, imputation_data, silent, max_num_samples
                )
                features, features_values, values = expl.feature_names, expl[0].data, expl[0].values
            elif self.shapley_alg == "Permutation":
                values, features_values, features = self.explain_event_permutation(
                    src, dst, timestamp, e_id, subgraphs_src, subgraphs_dst, imputation_data, silent
                )
            elif self.shapley_alg == "Exact":
                expl = self.explain_event_exact(
                    src, dst, timestamp, e_id, subgraphs_src, subgraphs_dst, imputation_data, silent
                )
                features, features_values, values = expl.feature_names, expl[0].data, expl[0].values
            else:
                raise NotImplementedError()

            # Append tuples for easier filtering later
            if self.node_names is not None:
                explanation_flatten.extend([
                    (e_id, self.node_names[s], self.node_names[d], label, timing, features[i],
                     features_values[i],
                     values[i])
                    for i, _ in enumerate(features)
                ])
            else:
                explanation_flatten.extend([
                    (e_id, s, d, label, timing, features[i], features_values[i], values[i])
                    for i, _ in enumerate(features)
                ])

        # Sort by absolute importance
        explanation_flatten = sorted(explanation_flatten, key=lambda x: np.abs(x[-1]), reverse=True)
        return explanation_flatten, remaining_ids, remaining_shapley_values, baseline

    def _get_labels(self, num_feat):
        """
        Get feature labels including structure & timing.
        """
        
        if type(self.feature_names) is not np.ndarray and self.feature_names == True:
            labels = ["Structure", "Timing"] + [str(i + 2) for i in range(num_feat)]
        elif self.feature_names is not None and self.feature_names is not False:
            labels = np.concat((np.array(["Structure", "Timing"]), self.feature_names))
        else:
            labels = ["0", "1"] + [str(i + 2) for i in range(num_feat)]
        return labels

    def explain_event_kernel(
        self, src: int, dst: int, timestamp: int, event_id: int,
        subgraphs_src: BatchSubgraphs, subgraphs_dst: BatchSubgraphs,
        imputation_data: dict, silent: bool
    ):
        """
        Explain the importance of *individual features* of a specific event using KernelSHAP in both steps.

        Parameters
        ----------
        src : int
            Source node ID of the prediction being explained.
        dst : int
            Destination node ID.
        timestamp : int
            Timestamp of the interaction.
        event_id : int
            Target event ID to explain.
        subgraphs_src : BatchSubgraphs
            Source-side local subgraph context.
        subgraphs_dst : BatchSubgraphs
            Destination-side local subgraph context.
        imputation_data : dict
            Dictionary mapping event IDs to feature default values for masking.
        silent : bool
            If True, disables SHAP's verbose output.

        Returns
        -------
        shap.Explanation
            SHAP explanation object for event features.
        """
        batch_size = CONFIG.shapley.batch_size
        use_cache = False    # flag for reusing predictions when masking features

        predictions = np.array([])  # stores predictions for repeated calls
        event_mask_mapping = {}
        feature_mask_mapping = {}
        fx = np.array([])  # model output for the original (unmasked) feature set

        # Retrieve all events in the subgraph (deduplicated)
        event_ids = np.concat([subgraphs_src.get_events(), subgraphs_dst.get_events()]).reshape((-1,))
        event_ids = np.unique(event_ids[event_ids != 0])

        def val_features(feature_mask: np.ndarray):
            """
            Inner SHAP function over *feature* masks for the specific event.
            """
            nonlocal use_cache

            def val_events(event_mask):
                """
                Inner function over *event masks*, called by SHAP while iterating over subsets.
                """
                nonlocal predictions, event_mask_mapping, feature_mask_mapping, fx

                if not use_cache:
                    # Predictions shape: (num_event_masks, num_feature_masks)
                    predictions = np.zeros((event_mask.shape[0], feature_mask.shape[0]))

                    # fx receives prediction for original, fully present events
                    fx = np.zeros((feature_mask.shape[0],)) if (event_mask != 0).all() else fx

                    # Prepare repeated subgraphs for given batch size
                    sg_src = copy.deepcopy(subgraphs_src)
                    sg_dst = copy.deepcopy(subgraphs_dst)
                    sg_src.repeat_nodes(event_mask.shape[0])
                    sg_dst.repeat_nodes(event_mask.shape[0])

                    # Mask events specified by event_mask
                    to_mask_events = event_mask == 0
                    sg_src.mask_events(event_ids, event_mask=to_mask_events, data_per_event=imputation_data)
                    sg_dst.mask_events(event_ids, event_mask=to_mask_events, data_per_event=imputation_data)

                    # Convert feature mask to tensors
                    feature_mask_torch = torch.from_numpy(feature_mask[:, 2:]).float().to(CONFIG.model.device)
                    node_attention = torch.from_numpy(feature_mask[:, 0]).float().to(CONFIG.model.device)

                    # Identify which rows affect the current event
                    per_batch_mask = np.full((event_mask.shape[0], 1), -1)
                    per_batch_mask[[np.isin(event_id, event_mask[i, :]) for i in range(event_mask.shape[0])]] = event_id

                    # Store mapping for lookup
                    event_mask_mapping = {tuple(event_mask[i]): i for i in range(event_mask.shape[0])}
                    feature_mask_mapping = {tuple(feature_mask[i]): i for i in range(feature_mask.shape[0])}

                    # Prepare subgraph copies for each feature mask in feature_mask
                    subgraphs_src_list = [copy.deepcopy(sg_src) for _ in range(feature_mask.shape[0])]
                    subgraphs_dst_list = [copy.deepcopy(sg_dst) for _ in range(feature_mask.shape[0])]

                    for i_sample in range(feature_mask.shape[0]):
                        # Apply masking to event's features and attentions
                        subgraphs_src_list[i_sample].mask_event_features(per_batch_mask, feature_mask_torch[i_sample, :])
                        subgraphs_dst_list[i_sample].mask_event_features(per_batch_mask, feature_mask_torch[i_sample, :])
                        subgraphs_src_list[i_sample].mask_node_attention(per_batch_mask, node_attention[i_sample])
                        subgraphs_dst_list[i_sample].mask_node_attention(per_batch_mask, node_attention[i_sample])
                        subgraphs_src_list[i_sample].mask_event_timing(per_batch_mask, feature_mask[i_sample, 1])
                        subgraphs_dst_list[i_sample].mask_event_timing(per_batch_mask, feature_mask[i_sample, 1])

                    # Run model in mini-batches combining different feature masks
                    num_feature_masks_per_batch = max(
                        1, int(batch_size / event_mask.shape[0])
                    )
                    for i_sample in range(0, feature_mask.shape[0], num_feature_masks_per_batch):
                        sg_src_batch = concat_subgraphs(subgraphs_src_list[i_sample:i_sample + num_feature_masks_per_batch])
                        sg_dst_batch = concat_subgraphs(subgraphs_dst_list[i_sample:i_sample + num_feature_masks_per_batch])

                        srcs = np.full((sg_src_batch.get_num_instances(),), src)
                        dsts = np.full((sg_src_batch.get_num_instances(),), dst)
                        time_stamps = np.full((sg_src_batch.get_num_instances(),), timestamp)

                        y = _predict_model_batchwise(
                            self.model, srcs, dsts, time_stamps,
                            sg_src_batch, sg_dst_batch
                        )
                        if CONFIG.model.task == "classification":
                            y = y.sigmoid()

                        y = y.cpu().detach().numpy().reshape((event_mask.shape[0], -1), order="F")
                        predictions[:, i_sample:i_sample + num_feature_masks_per_batch] = y

                        # If all events are present, store this baseline prediction in fx
                        if (event_mask != 0).all():
                            fx[i_sample:i_sample + num_feature_masks_per_batch] = y

                    # Return ordered predictions matching SHAP's current (event_mask, feature_mask) pairing
                    if feature_mask.shape[0] != 1:
                        result = np.array([predictions[event_mask_mapping[tuple(event_mask[i])], feature_mask_mapping[tuple(fm)]]
                                           for i, fm in enumerate(feature_mask)])
                    else:
                        result = np.array([predictions[event_mask_mapping[tuple(event_mask[i])],
                                                       feature_mask_mapping[tuple(feature_mask[0])]]
                                           for i in range(event_mask.shape[0])])
                    return result
                else:
                    # Use cached predictions on subsequent calls
                    return np.array([predictions[event_mask_mapping[tuple(event_mask[i])], feature_mask_mapping[tuple(fm)]]
                                     for i, fm in enumerate(feature_mask)])

            # Second-level KernelSHAP call: apply over event subsets for given feature_mask
            result = np.zeros((feature_mask.shape[0]))
            use_cache = False
            explainer = shap.explainers.KernelExplainer(val_events, data=np.zeros((1, len(event_ids))))
            explainer(np.array([event_ids]).reshape(1, -1), silent=silent)  # prime cache
            use_cache = True

            # For each feature mask, retrieve SHAP value for this event
            for i in range(feature_mask.shape[0]):
                explainer.fx = fx[i:i + 1]  # baseline output for given feature mask
                explainer.nsamplesRun = 0
                explainer.run()
                phi = np.zeros((explainer.data.groups_size, explainer.D))
                phi_var = np.zeros_like(phi)
                for d in range(explainer.D):
                    vphi, vphi_var = explainer.solve(explainer.nsamples / explainer.max_samples, d)
                    phi[explainer.varyingInds, d] = vphi
                    phi_var[explainer.varyingInds, d] = vphi_var
                if not explainer.vector_out:
                    phi = np.squeeze(phi, axis=1)
                    phi_var = np.squeeze(phi_var, axis=1)
                result[i] = phi[event_ids == event_id]
            return result

        # Gather event features and labels for SHAP
        event_features = self.neighbor_finder.get_edge_features(np.array([event_id])).detach().numpy().flatten()
        labels = self._get_labels(event_features.shape[0])

        explainer = shap.explainers.KernelExplainer(
            val_features,
            data=imputation_data[event_id].cpu().numpy().reshape(1, -1),
            feature_names=labels
        )
        if CONFIG.model.task == "classification": 
            d = np.concat(([1, self.data.node_interact_times[event_id]], event_features))
        else:
            d = np.concat(([1, self.data.node_interact_times[event_id]], event_features))
        return explainer(d.reshape(1, -1), silent=silent)

    def get_y_ref(
        self, src: int, dst: int, event_ids: np.ndarray, timestamp: int, k: int,
        subgraph_src: BatchSubgraphs, subgraph_dst: BatchSubgraphs,
        event_permutations: np.ndarray, imputation_data: dict
    ):
        """
        Compute baseline predictions for coalitions **without** the target event(s) present.

        This function takes a batch of event masks (event_permutations) where
        the target event(s) to evaluate have been removed (set to zero).
        It applies these masks to cloned source/destination subgraphs, replaces
        the masked events with imputed values, and runs the GNN to obtain predictions.

        Parameters
        ----------
        src : int
            Source node ID.
        dst : int
            Destination node ID.
        event_ids : np.ndarray
            Array of event IDs in the local subgraph.
        timestamp : int
            Interaction timestamp.
        k : int
            Number of coalition masks to evaluate (batch size).
        subgraph_src : BatchSubgraphs
            Source-side local subgraph.
        subgraph_dst : BatchSubgraphs
            Destination-side local subgraph.
        event_permutations : np.ndarray
            Binary/batch mask array indicating which events to keep (0 = mask).
        imputation_data : dict
            Mapping from eeventdge ID to imputation feature tensor.

        Returns
        -------
        np.ndarray
            Predictions reshaped to `(1, k)` for the reference coalitions (no target event).
        """
        srcs = np.full((k,), src)
        dsts = np.full((k,), dst)
        time_stamps = np.full((k,), timestamp)
        sg_src = copy.deepcopy(subgraph_src)
        sg_dst = copy.deepcopy(subgraph_dst)
        sg_src.repeat_nodes(k)
        sg_dst.repeat_nodes(k)
        to_mask_events = event_permutations == 0
        sg_src.mask_events(event_ids, event_mask=to_mask_events, data_per_event=imputation_data)
        sg_dst.mask_events(event_ids, event_mask=to_mask_events, data_per_event=imputation_data)
        y_ref = _predict_model_batchwise(
            self.model, srcs, dsts, time_stamps, sg_src, sg_dst
        )
        if CONFIG.model.task == "classification":
            y_ref = y_ref.sigmoid()
        return y_ref.reshape((1, -1))


    def get_y(
        self, src: int, dst: int, timestamp: int, event_ids: np.ndarray, event_id: int,
        k: int, l: int, subgraph_src: BatchSubgraphs, subgraph_dst: BatchSubgraphs,
        event_permutations: np.ndarray, feature_mask: np.ndarray, imputation_data: dict
    ):
        """
        Compute predictions for coalitions **including** the target event, 
        with various feature masks applied to that event.

        Parameters
        ----------
        src : int
            Source node ID.
        dst : int
            Destination node ID.
        timestamp : int
            Interaction timestamp.
        event_ids : np.ndarray
            Array of all event IDs in the subgraph.
        event_id : int
            Target event ID whose features are being varied.
        k : int
            Number of coalition permutations for events.
        l : int
            Number of feature masks for the target event.
        subgraph_src : BatchSubgraphs
            Source-side subgraph.
        subgraph_dst : BatchSubgraphs
            Destination-side subgraph.
        event_permutations : np.ndarray
            Edge inclusion masks per coalition (0 = mask).
        feature_mask : np.ndarray
            Array of shape `(l, num_features+2)` specifying feature/timing/node-attention settings.
        imputation_data : dict
            Mapping from event ID to default feature values for imputation.

        Returns
        -------
        torch.Tensor
            Model predictions of shape `(l, k)` for each feature mask and coalition.
        """
        srcs = np.full((k,), src)
        dsts = np.full((k,), dst)
        time_stamps = np.full((k,), timestamp)
        sg_src = copy.deepcopy(subgraph_src)
        sg_dst = copy.deepcopy(subgraph_dst)
        sg_src.repeat_nodes(k)
        sg_dst.repeat_nodes(k)
        to_mask_events = event_permutations == 0
        sg_src.mask_events(event_ids, event_mask=to_mask_events, data_per_event=imputation_data)
        sg_dst.mask_events(event_ids, event_mask=to_mask_events, data_per_event=imputation_data)
        feature_mask_torch = torch.from_numpy(feature_mask).float().to(CONFIG.model.device)
        default_time_stamps = feature_mask_torch[:, 1].detach().cpu().numpy()
        y = torch.zeros((l, k))
        event_masks_src = sg_src.get_event_masks(event_id)
        event_masks_dst = sg_dst.get_event_masks(event_id)
        for i, mask in enumerate(feature_mask_torch):
            # Replace features for event in the subgraph with masked values
            sg_src.replace_event(event_masks_src, mask[0], default_time_stamps[i], mask[2:])
            sg_dst.replace_event(event_masks_dst, mask[0], default_time_stamps[i], mask[2:])
            # Forward pass
            y[i, :] = _predict_model_batchwise(
                self.model, srcs, dsts, time_stamps, sg_src, sg_dst
            ).reshape(-1).detach().cpu()
            if CONFIG.model.task == "classification":
                y[i, :] = y[i, :].sigmoid()
        return y
    
    def explain_event_exact(
        self, src: int, dst: int, timestamp: int, event_id: int,
        subgraphs_src: BatchSubgraphs, subgraphs_dst: BatchSubgraphs,
        imputation_data: dict, silent: bool
    ):
        event_ids = np.concat([subgraphs_src.get_events(), subgraphs_dst.get_events()]).reshape((-1,))
        event_ids = np.unique(event_ids[event_ids != 0])
        event_explainer = ShapleyExplainerEvents(self.model, self.neighbor_finder, self.data, self.event_features, algorithm="Exact") 

        def val_features(feature_mask: np.ndarray):
            default_time_stamps = feature_mask[:, 1]
            phis = np.zeros((feature_mask.shape[0],))
            for i, m in enumerate(feature_mask):
                event_masks_src = subgraphs_src.get_event_masks(event_id)
                event_masks_dst = subgraphs_dst.get_event_masks(event_id)
                subgraphs_src.replace_event(event_masks_src, torch.tensor(m[0], dtype=torch.float32).to(CONFIG.model.device), default_time_stamps[i], torch.tensor(m[2:], dtype=torch.float32).to(CONFIG.model.device))
                subgraphs_dst.replace_event(event_masks_dst, torch.tensor(m[0], dtype=torch.float32).to(CONFIG.model.device), default_time_stamps[i], torch.tensor(m[2:], dtype=torch.float32).to(CONFIG.model.device))

                shap_values = event_explainer._explain_instance(src, dst, timestamp, subgraphs_src, subgraphs_dst, event_ids, imputation_data, silent)[1]
                phi = shap_values.values[0][event_ids == event_id]
                phis[i] = phi
            return phis

        event_features = self.neighbor_finder.get_edge_features(np.array([event_id])).detach().numpy().flatten()
        labels = self._get_labels(event_features.shape[0])
        explainer = shap.ExactExplainer(
            val_features, masker=imputation_data[event_id].cpu().numpy().reshape(1, -1), feature_names=labels
        )
        if CONFIG.model.task == "classification":
            d = np.concat(([1, self.data.node_interact_times[event_id]], event_features))
        else:
            d = np.concat(([1, self.data.node_interact_times[event_id]], event_features))
        return explainer(d.reshape(1, -1), silent=silent)

    def explain_event_monte_carlo(
        self, src: int, dst: int, timestamp: int, event_id: int,
        subgraphs_src: BatchSubgraphs, subgraphs_dst: BatchSubgraphs,
        imputation_data: dict, silent: bool, max_num_samples: Optional[int] = None
    ):
        """
        Estimate event-feature Shapley values using Monte Carlo sampling.

        This method approximates Shapley values by randomly permuting the order of event inclusion,
        calculating differences in model output with and without the target event, and averaging.

        Parameters
        ----------
        src, dst : int
            Source and destination node IDs.
        timestamp : int
            Time of interaction for which to explain the prediction.
        event_id : int
            Target event ID for feature-level explanation.
        subgraphs_src, subgraphs_dst : BatchSubgraphs
            Source and destination-side local subgraphs.
        imputation_data : dict
            Default feature/timing values for masked events.
        silent : bool
            Whether to suppress SHAP progress output.
        max_num_samples : Optional[int]
            Maximum number of permutation samples. If omitted, the value from
            ``Shapley.max_num_samples`` is used.

        Returns
        -------
        shap.Explanation
            SHAP explanation object containing feature attributions for the target event.
        """
        event_ids = np.concat([subgraphs_src.get_events(), subgraphs_dst.get_events()]).reshape((-1,))
        event_ids = np.unique(event_ids[event_ids != 0])
        if max_num_samples is None:
            max_num_samples = CONFIG.shapley.max_num_samples

        def val_features(feature_mask: np.ndarray):
            k = min(len(event_ids) * 5, max_num_samples)
            event_permutations = np.tile(event_ids, (k, 1))
            event_permutations = np.apply_along_axis(np.random.permutation, axis=1, arr=event_permutations)
            pos_event = np.where(event_permutations == event_id)[1]
            event_permutations_with_event = event_permutations.copy()
            for i, p in enumerate(pos_event):
                event_permutations[i, p:] = 0
                event_permutations_with_event[i, (p + 1):] = 0
            event_mask = np.tile(event_ids, (k, 1))
            event_mask[~((event_mask[:, :, None] == event_permutations[:, None, :]).any(axis=-1))] = 0
            event_mask_with_event = np.tile(event_ids, (k, 1))
            event_mask_with_event[~((event_mask_with_event[:, :, None] == event_permutations_with_event[:, None, :]).any(axis=-1))] = 0
            y_ref = self.get_y_ref(src, dst, event_ids, timestamp, k, subgraphs_src, subgraphs_dst, event_mask, imputation_data)
            y = self.get_y(src, dst, timestamp, event_ids, event_id, k, feature_mask.shape[0],
                        subgraphs_src, subgraphs_dst, event_mask_with_event, feature_mask, imputation_data)
            phi = torch.mean((y.cpu() - y_ref.cpu()), dim=1, keepdim=True).detach().numpy()
            return phi.reshape((-1,))

        event_features = self.neighbor_finder.get_edge_features(np.array([event_id])).detach().numpy().flatten()
        labels = self._get_labels(event_features.shape[0])
        explainer = shap.explainers.KernelExplainer(
            val_features, data=imputation_data[event_id].cpu().numpy().reshape(1, -1), feature_names=labels
        )
        if CONFIG.model.task == "classification": 
            d = np.concat(([1, self.data.node_interact_times[event_id]], event_features))
        else:
            d = np.concat(([1, self.data.node_interact_times[event_id]], event_features))
        return explainer(d.reshape(1, -1), silent=silent, l1_reg=False)


    def explain_event_permutation(
        self, src: int, dst: int, timestamp: int, event_id: int,
        subgraphs_src: BatchSubgraphs, subgraphs_dst: BatchSubgraphs,
        imputation_data: dict, silent: bool
    ):
        """
        Explain event feature importance using permutation-based masking (Owen values).

        This method iteratively masks/unmasks each feature of the target event
        while holding others random, and measures the mean change in prediction.

        Parameters
        ----------
        src, dst : int
            Source and destination node IDs.
        timestamp : int
            Time of interaction being explained.
        event_id : int
            Target event ID for feature-level examination.
        subgraphs_src, subgraphs_dst : BatchSubgraphs
            Source and destination local subgraphs.
        imputation_data : dict
            Default values for feature/timing imputation when masked.
        silent : bool
            (Unused) Passed for API consistency with other explainers.

        Returns
        -------
        tuple
            - owen_vals : np.ndarray — Mean marginal contribution for each feature.
            - event_features_torch[0, :] : np.ndarray — Original feature vector (including structure/timing).
            - labels : list — Human-readable feature names.
        """
        k = 100
        event_ids = np.unique(np.concat([subgraphs_src.get_events(), subgraphs_dst.get_events()]).reshape((-1,)))
        event_ids = event_ids[event_ids != 0]
        event_permutations = np.tile(event_ids, (k, 1))
        event_permutations = np.apply_along_axis(np.random.permutation, axis=1, arr=event_permutations)
        pos_event = np.where(event_permutations == event_id)[1]
        for i, p in enumerate(pos_event):
            event_permutations[i, (p + 1):] = 0
        event_mask = np.tile(event_ids, (k, 1))
        event_mask[~((event_mask[:, :, None] == event_permutations[:, None, :]).any(axis=-1))] = 0
        sg_src = copy.deepcopy(subgraphs_src)
        sg_dst = copy.deepcopy(subgraphs_dst)
        sg_src.repeat_nodes(k)
        sg_dst.repeat_nodes(k)
        sg_src.mask_events(event_ids, event_mask, imputation_data)
        sg_dst.mask_events(event_ids, event_mask, imputation_data)
        event_id_masks_src = sg_src.get_event_masks(event_id)
        event_id_masks_dst = sg_dst.get_event_masks(event_id)
        event_features = self.neighbor_finder.get_edge_features(np.array([event_id])).flatten()
        features = np.arange(event_features.shape[0] + 2)
        # Randomly permute subset of features
        feature_permutations = np.zeros((k, len(features)))
        for i in range(k):
            j = np.random.randint(1, len(features))
            feature_permutations[i, :j] = np.random.choice(a=features, size=(j,), replace=False)
        feat_mask = np.ones_like(feature_permutations, dtype=bool)
        feat_mask[~((features[None, :, None] == feature_permutations[:, None, :]).any(axis=-1))] = False
        feat_mask = torch.from_numpy(feat_mask).to(CONFIG.model.device)
        event_features_torch = torch.concat((torch.tensor([1, timestamp]), event_features))
        event_features_torch = event_features_torch.reshape(1, -1).repeat(k, 1).to(CONFIG.model.device)
        default_values = imputation_data[event_id].reshape(1, -1).repeat(k, 1).to(CONFIG.model.device)
        srcs = np.full((k,), src)
        dsts = np.full((k,), dst)
        time_stamps = np.full((k,), timestamp)
        owen_vals = np.zeros_like(features, dtype="float32")
        for f in features:
            # Mask feature f (negative case) and unmask (positive case)
            feat_mask_neg = feat_mask.clone()
            feat_mask_pos = feat_mask.clone()
            feat_mask_neg[:, f] = False
            feat_mask_pos[:, f] = True
            feat_neg = torch.where(feat_mask_neg, event_features_torch, default_values)
            feat_pos = torch.where(feat_mask_pos, event_features_torch, default_values)
            sg_src.replace_event_2D(event_id_masks_src, feat_neg[:, 0], feat_neg[:, 1].cpu().numpy(), feat_neg[:, 2:])
            sg_dst.replace_event_2D(event_id_masks_dst, feat_neg[:, 0], feat_neg[:, 1].cpu().numpy(), feat_neg[:, 2:])
            y_neg = _predict_model_batchwise(
                self.model, srcs, dsts, time_stamps, sg_src, sg_dst
            )
            sg_src.replace_event_2D(event_id_masks_src, feat_pos[:, 0], feat_pos[:, 1].cpu().numpy(), feat_pos[:, 2:])
            sg_dst.replace_event_2D(event_id_masks_dst, feat_pos[:, 0], feat_pos[:, 1].cpu().numpy(), feat_pos[:, 2:])
            y_pos = _predict_model_batchwise(
                self.model, srcs, dsts, time_stamps, sg_src, sg_dst
            )
            owen_vals[f] = torch.mean(y_pos - y_neg).detach().cpu().item()
        return owen_vals, event_features_torch[0, :].detach().cpu().numpy(), self._get_labels(event_features.shape[0])