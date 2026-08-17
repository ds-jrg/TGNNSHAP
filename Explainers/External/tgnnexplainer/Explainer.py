"""DyGLib integration for T-GNNExplainer/SubgraphX.

The upstream project targets its own TGAT/TGN implementations.  This adapter
keeps its event-level MCTS explorer and paper navigator while translating model
calls and temporal neighborhoods to DyGLib.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from Config.config import CONFIG
from DyGLib.models.modules import TGNN
from DyGLib.utils.DataLoader import Data
from DyGLib.utils.utils import BatchSubgraphs, NeighborSampler
from Explainers.utils import Explainer

# The upstream package uses absolute imports rooted at ``tgnnexplainer``.
_PACKAGE_ROOT = Path(__file__).resolve().parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from tgnnexplainer.xgraph.method.navigators import MLPNavigator
from tgnnexplainer.xgraph.method.other_baselines_tg import PGExplainerExt
from tgnnexplainer.xgraph.method.subgraphx_tg import SubgraphXTG


CONFIG = CONFIG()


class _DyGLibNeighborAdapter:
    def __init__(self, sampler: NeighborSampler):
        self.sampler = sampler

    def get_temporal_neighbor(self, node_ids, times, num_neighbors, edge_idx_preserve_list=None):
        kept = None
        if edge_idx_preserve_list is not None:
            kept = np.repeat(np.asarray(edge_idx_preserve_list, dtype=np.int64)[None, :], len(node_ids), axis=0)
        return self.sampler.get_historical_neighbors(
            np.asarray(node_ids), np.asarray(times), num_neighbors=num_neighbors,
            kept_edge_ids=kept,
        )


class _DyGLibModelFacade:
    """Expose the small TGAT API used by the upstream explorer/navigator."""
    def __init__(self, model: TGNN, sampler: NeighborSampler, events: pd.DataFrame,
                 raw_node_by_normalized: dict[int, int]):
        self._model = model
        self._sampler = sampler
        self.raw_node_by_normalized = raw_node_by_normalized
        self.ngh_finder = _DyGLibNeighborAdapter(sampler)
        self.num_layers = CONFIG.model.num_layers
        self.num_neighbors = CONFIG.model.num_neighbors
        raw_features = model.backbone.node_features.detach()
        self.node_raw_embed = raw_features.new_zeros((max(raw_node_by_normalized) + 1, raw_features.shape[1]))
        for normalized, raw in raw_node_by_normalized.items():
            self.node_raw_embed[normalized] = raw_features[raw]
        self.edge_raw_embed = sampler.edge_features.detach()
        self.time_encoder = model.backbone.time_encoder
        self.model_name = CONFIG.model.model_name.lower()
        self.model_dim = max(1, int((2 * self.node_raw_embed.shape[1] +
                                     self.time_encoder.time_dim + self.edge_raw_embed.shape[1]) / 4))
        self.explainer_input_dim = 2 * (2 * self.node_raw_embed.shape[1] +
                                        self.time_encoder.time_dim + self.edge_raw_embed.shape[1])
        self.n_node_features = max(1, self.explainer_input_dim // 8)
        self.all_events = events

    def eval(self):
        self._model.eval()
        return self

    def to(self, device):
        self._model.to(device)
        self.node_raw_embed = self.node_raw_embed.to(device)
        self.edge_raw_embed = self.edge_raw_embed.to(device)
        return self

    def _subgraphs(self, src, dst, times, kept=None, candidate_weights=None):
        src = np.asarray([self.raw_node_by_normalized[int(x)] for x in src], dtype=np.int64)
        dst = np.asarray([self.raw_node_by_normalized[int(x)] for x in dst], dtype=np.int64)
        src_data = self._sampler.get_multi_hop_neighbors(
            self.num_layers, src, times, self.num_neighbors, kept_edge_ids=kept)
        dst_data = self._sampler.get_multi_hop_neighbors(
            self.num_layers, dst, times, self.num_neighbors, kept_edge_ids=kept)
        src_features = self._sampler.get_edge_features_for_multi_hop(src_data[1])
        dst_features = self._sampler.get_edge_features_for_multi_hop(dst_data[1])
        src_sg = BatchSubgraphs(*src_data, src_features)
        dst_sg = BatchSubgraphs(*dst_data, dst_features)
        if candidate_weights is not None:
            candidate_ids, weights = candidate_weights
            candidate_ids = np.asarray(candidate_ids)
            weights = torch.as_tensor(weights, dtype=torch.float32, device=self._model.backbone.device).reshape(-1)
            for layer, events in enumerate(src_sg.events):
                attention = torch.ones(events.shape, device=weights.device)
                for event_id, weight in zip(candidate_ids, weights):
                    attention[torch.from_numpy(events == event_id).to(weights.device)] = weight
                src_sg.event_attention[layer] = attention
            for layer, events in enumerate(dst_sg.events):
                attention = torch.ones(events.shape, device=weights.device)
                for event_id, weight in zip(candidate_ids, weights):
                    attention[torch.from_numpy(events == event_id).to(weights.device)] = weight
                dst_sg.event_attention[layer] = attention
        src_sg.to(self._model.backbone.device)
        dst_sg.to(self._model.backbone.device)
        return src_sg, dst_sg

    def get_prob(self, src, dst, times, logit=True, candidate_weights_dict=None,
                 edge_idx_preserve_list=None, **kwargs):
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        times = np.asarray(times, dtype=np.float64)
        kept = None
        if edge_idx_preserve_list is not None:
            kept = np.repeat(np.asarray(edge_idx_preserve_list, dtype=np.int64)[None, :], len(src), axis=0)
        candidate_weights = None
        if candidate_weights_dict is not None:
            candidate_weights = (candidate_weights_dict["candidate_events"].detach().cpu().numpy(),
                                 candidate_weights_dict["edge_weights"])
        src_sg, dst_sg = self._subgraphs(src, dst, times, kept, candidate_weights)
        output = self._model(src, dst, times, src_sg, dst_sg, edges_are_positive=False)
        return output if logit else output.sigmoid()


class _DyGLibSubgraphXTG(SubgraphXTG):
    """Use DyGLib neighbor sampling while retaining the paper's MCTS logic."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = CONFIG.model.num_layers
        self.num_neighbors = CONFIG.model.num_neighbors

    def find_candidates(self, target_event_idx):
        row = self.all_events.iloc[int(target_event_idx) - 1]
        roots = np.array([
            self.model.raw_node_by_normalized[int(row.u)],
            self.model.raw_node_by_normalized[int(row.i)],
        ], dtype=np.int64)
        times = np.array([row.ts, row.ts], dtype=np.float64)
        seen = []
        current_nodes, current_times = roots, times
        for _ in range(self.num_layers):
            nodes, events, event_times = self.model.ngh_finder.get_temporal_neighbor(
                current_nodes, current_times, self.num_neighbors,
                edge_idx_preserve_list=self.ori_subgraph_df.e_idx.to_numpy(),
            )
            if not np.any(events):
                nodes, events, event_times = self.model.ngh_finder.get_temporal_neighbor(
                    current_nodes, current_times, self.num_neighbors,
                )
            seen.extend(events.reshape(-1).tolist())
            current_nodes = nodes.reshape(-1)
            current_times = event_times.reshape(-1)
        unique = sorted(set(int(x) for x in seen if int(x) != 0))
        candidates = unique[-self.threshold_num:] if len(unique) > self.threshold_num else unique
        return candidates, unique


class SubgraphXTExplainer(Explainer):
    def __init__(self, model: TGNN, neighbor_finder: NeighborSampler, data: Data):
        super().__init__(model, neighbor_finder, data)
        if CONFIG.model.model_name not in ("TGAT", "TGN"):
            raise NotImplementedError(
                "T-GNNExplainer currently supports DyGLib TGAT and TGN backbones only"
            )
        self.events = self._make_events(data)
        self.original_event_ids = data.edge_ids.astype(np.int64)
        raw_nodes = np.unique(np.concatenate([data.src_node_ids, data.dst_node_ids]))
        normalized_nodes = np.concatenate([
            self.events.u.to_numpy(dtype=np.int64),
            self.events.i.to_numpy(dtype=np.int64),
        ])
        self.raw_node_by_normalized = {
            int(normalized): int(raw)
            for normalized, raw in zip(normalized_nodes, np.concatenate([data.src_node_ids, data.dst_node_ids]))
        }
        self.facade = _DyGLibModelFacade(
            model, neighbor_finder, self.events, self.raw_node_by_normalized
        )
        self.explainer = None

    @staticmethod
    def _navigator_checkpoint_path():
        checkpoint_dir = Path("Saved_models") / CONFIG.data.dataset_name / "tgnnexplainer"
        return PGExplainerExt._ckpt_path(
            checkpoint_dir,
            CONFIG.model.model_name.lower(),
            CONFIG.data.dataset_name,
            "subgraphx_tg",
        )

    @staticmethod
    def train_model_if_missing(model: TGNN, neighbor_finder: NeighborSampler, data: Data):
        """Train the navigator on training data, or reuse its checkpoint."""
        checkpoint_path = SubgraphXTExplainer._navigator_checkpoint_path()
        if checkpoint_path.exists():
            print(f"Using cached T-GNNExplainer navigator: {checkpoint_path}")
            return

        print("T-GNNExplainer navigator is missing; training navigator...")
        preparer = SubgraphXTExplainer(model, neighbor_finder, data)
        preparer._create_navigator()
        print(f"Cached T-GNNExplainer navigator: {checkpoint_path}")

    # Backward-compatible alias for callers that used the former name.
    preprocess = train_model_if_missing

    @staticmethod
    def _make_events(data: Data) -> pd.DataFrame:
        events = data.dataset.copy() if data.dataset is not None else pd.DataFrame()
        events = events.copy()
        source_nodes = np.unique(data.src_node_ids)
        destination_nodes = np.unique(data.dst_node_ids)
        source_map = {int(node): i + 1 for i, node in enumerate(source_nodes)}
        destination_offset = len(source_map)
        destination_map = {int(node): destination_offset + i + 1 for i, node in enumerate(destination_nodes)}
        events["u"] = np.array([source_map[int(node)] for node in data.src_node_ids], dtype=np.int64)
        events["i"] = np.array([destination_map[int(node)] for node in data.dst_node_ids], dtype=np.int64)
        events["ts"] = data.node_interact_times.astype(np.float64)
        events["e_idx"] = np.arange(1, len(data.edge_ids) + 1, dtype=np.int64)
        events["idx"] = events["e_idx"]
        events["label"] = data.labels
        columns = ["u", "i", "ts", "label", "e_idx", "idx"]
        remaining = [column for column in events.columns if column not in columns]
        return events[columns + remaining]

    def _create_navigator(self):
        dataset = CONFIG.data.dataset_name
        model_name = CONFIG.model.model_name.lower()
        params = CONFIG.tgnnExplainerConfig
        checkpoint_dir = Path("Saved_models") / CONFIG.data.dataset_name / "tgnnexplainer"
        results_dir = Path("Logs/TGNNExplainer") / dataset
        results_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # The paper's navigator is the pairwise MLP navigator, not the
        # authors' PGNavigator variant.
        navigator = MLPNavigator(
            self.facade, model_name, "subgraphx_tg", dataset, self.events, "event",
            device=CONFIG.model.device, results_dir=str(results_dir), debug_mode=False,
            train_epochs=getattr(CONFIG.tgnnExplainerConfig, "navigator_train_epochs", 50),
            explainer_ckpt_dir=str(checkpoint_dir),
            reg_coefs=(0.5, 0.1), batch_size=16, lr=1e-4,
        )
        self.explainer = _DyGLibSubgraphXTG(
            self.facade, model_name, "subgraphx_tg", dataset, self.events, "event",
            device=CONFIG.model.device, results_dir=str(results_dir), debug_mode=False,
            threshold_num=20, save_results=False, mcts_saved_dir=str(results_dir),
            load_results=False, rollout=params.num_rollouts, min_atoms=params.min_atoms,
            c_puct=5, navigator=navigator, navigator_type="mlp", pg_positive=True,
        )

    def initialize(self):
        checkpoint_path = SubgraphXTExplainer._navigator_checkpoint_path()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"T-GNNExplainer navigator is missing: {checkpoint_path}. "
                "Call SubgraphXTExplainer.train_model_if_missing() first."
            )
        self._create_navigator()
        print(f"Using cached T-GNNExplainer navigator: {checkpoint_path}")

    def explain_instance(self, src: int, dst: int, timestamp: float, silent: bool = False) -> Any:
        if self.explainer is None:
            raise RuntimeError("Call initialize() before explaining events")
        matches = np.flatnonzero(
            (self.data.src_node_ids == src) &
            (self.data.dst_node_ids == dst) &
            (self.data.node_interact_times == timestamp)
        )
        if len(matches) == 0:
            raise KeyError(f"No event found for ({src}, {dst}, {timestamp})")
        event_idx = int(matches[0]) + 1
        self.explainer.debug_mode = not silent
        self.explainer.verbose = not silent
        tree_nodes, best_node = self.explainer(event_idxs=event_idx)[0]
        candidate_scores = getattr(self.explainer, "candidate_initial_weights", {})
        ranked = list(best_node.coalition)
        ranked.extend(sorted(
            (e for e in self.explainer.candidate_events if e not in ranked),
            key=lambda e: candidate_scores.get(e, 0.0), reverse=True,
        ))
        ranked.extend(e for e in self.explainer.base_events if e not in ranked)
        original_ids = self.original_event_ids
        ranked_original = [int(original_ids[e - 1]) for e in ranked]
        # The shared evaluator samples DyGLib neighborhoods independently.
        # Add those exact two-hop events so the final coalition contains all
        # events in its unmasked reference subgraph.
        raw_src = int(self.data.src_node_ids[matches[0]])
        raw_dst = int(self.data.dst_node_ids[matches[0]])
        sampled = self.neighbor_finder.get_multi_hop_neighbors(
            CONFIG.model.num_layers,
            np.array([raw_src, raw_dst]),
            np.array([timestamp, timestamp]),
            num_neighbors=CONFIG.model.num_neighbors,
        )
        sampled_events = np.unique(np.concatenate(sampled[1], axis=1))
        ranked_original.extend(int(event_id) for event_id in sampled_events if event_id != 0)
        return np.asarray(list(dict.fromkeys(ranked_original)), dtype=np.int64)

    def build_coalitions(self, explanation):
        ranked = np.asarray(explanation, dtype=np.int64)
        ranked = np.unique(ranked)
        if len(ranked) == 0:
            return np.zeros((1, 1), dtype=np.int64), None, None
        coalitions = np.zeros((len(ranked), len(ranked)), dtype=np.int64)
        for i in range(len(ranked)):
            coalitions[i, :i + 1] = ranked[:i + 1]
        return coalitions, None, None
