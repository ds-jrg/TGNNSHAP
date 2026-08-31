"""DyGLib adapter for the QIEA-TGX temporal graph explainer.

The QIEA implementation is kept unchanged.  This module supplies the model
and neighborhood API it expects and exposes the repository's common
``Explainer`` interface.
"""
from __future__ import annotations

import sys
import importlib.util
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from Config.config import CONFIG
from DyGLib.models.modules import TGNN
from DyGLib.utils.DataLoader import Data
from DyGLib.utils.utils import BatchSubgraphs, NeighborSampler
from Explainers.utils import Explainer

_QIEA_ROOT = Path(__file__).resolve().parent
if str(_QIEA_ROOT) not in sys.path:
    sys.path.insert(0, str(_QIEA_ROOT))

from Xmethods.codes.functions.QIEA_func import Quantum_individual_optimizationA

CONFIG = CONFIG()

def _observation_strategy(func_index: int) -> str:
    """Translate the upstream numeric function selector to its algorithm name."""
    return {
        0: "none",
        1: "ob_hops",
        2: "ob_half",
        3: "ob_hop_timeXtime",
        4: "ob_hop_timeXtime_half",
        5: "ob_hopXtime",
        6: "ob_hop_all",
    }.get(int(func_index), "none")


class _QIEAModelFacade:
    """Adapt DyGLib's TGNN forward pass to QIEA-TGX's ``get_prob`` API."""

    def __init__(self, model: TGNN, sampler: NeighborSampler):
        self.model = model
        self.sampler = sampler
        self.num_layers = CONFIG.model.num_layers
        self.num_neighbors = CONFIG.model.num_neighbors

    def get_prob(self, src, dst, times, edge_idx_preserve_list=None, **_kwargs):
        src = np.asarray(src, dtype=np.int64).reshape(-1)
        dst = np.asarray(dst, dtype=np.int64).reshape(-1)
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        kept = None
        if edge_idx_preserve_list is not None:
            allowed = np.asarray(edge_idx_preserve_list, dtype=np.int64).reshape(-1)
            kept = np.repeat(allowed[None, :], len(src), axis=0)

        src_data = self.sampler.get_multi_hop_neighbors(
            self.num_layers, src, times, self.num_neighbors, kept_edge_ids=kept
        )
        dst_data = self.sampler.get_multi_hop_neighbors(
            self.num_layers, dst, times, self.num_neighbors, kept_edge_ids=kept
        )
        src_features = self.sampler.get_edge_features_for_multi_hop(src_data[1])
        dst_features = self.sampler.get_edge_features_for_multi_hop(dst_data[1])
        src_graph = BatchSubgraphs(*src_data, src_features)
        dst_graph = BatchSubgraphs(*dst_data, dst_features)
        src_graph.to(CONFIG.model.device)
        dst_graph.to(CONFIG.model.device)
        with torch.no_grad():
            logits = self.model(
                src_node_ids=src,
                dst_node_ids=dst,
                node_interact_times=times,
                src_subgraphs=src_graph,
                dst_subgraphs=dst_graph,
                time_gap=CONFIG.model.time_gap,
                edges_are_positive=False,
            )
        return logits.reshape(-1).sigmoid()


class QIEATGXExplainer(Explainer):
    """Quantum-inspired event explainer compatible with DyGLib evaluation."""

    def __init__(self, model: TGNN, neighbor_finder: NeighborSampler, data: Data, sparse_ratio: Optional[float] = None):
        super().__init__(model, neighbor_finder, data)
        if CONFIG.model.task.lower() not in {"classification", "link prediction", "link_prediction"}:
            raise NotImplementedError("QIEA-TGX currently supports link prediction only")
        if data.dataset is None:
            raise ValueError("QIEA-TGX requires Data.dataset for event lookup")
        self.device = torch.device(CONFIG.model.device)
        self.facade = _QIEAModelFacade(model, neighbor_finder)
        self._candidates: np.ndarray | None = None
        self.sparse_ratio = sparse_ratio

    def initialize(self):
        """QIEA has no separate training stage in this integration."""
        return None

    def _find_candidates(self, src: int, dst: int, timestamp: float):
        """Collect the same temporal computational neighborhood used by TGNN."""
        roots = np.array([src, dst], dtype=np.int64)
        times = np.array([timestamp, timestamp], dtype=np.float64)
        _, events, _ = self.neighbor_finder.get_multi_hop_neighbors(
            CONFIG.model.num_layers, roots, times, CONFIG.model.num_neighbors
        )
        candidates = np.unique(np.concatenate(events, axis=None))
        candidates = candidates[candidates != 0].astype(np.int64)
        
        assert events[0].shape[0] == 2 #One for src and one for dst
        hops = np.concatenate([
            events[0].reshape(-1, 1),
            events[1].reshape(-1, CONFIG.model.num_neighbors)
        ], axis=1)
         
        assert np.all(hops[hops[:,0] == 0, 1:] == 0), "The first column of hops should be the root nodes, and the rest should be their neighbors. If the first column is 0, then the rest should also be 0."
        hops = hops[hops[:,0] != 0, :] 
        return candidates, hops

    def explain_instance(self, src: int, dst: int, timestamp: float, silent: bool = False) -> Any:
        candidates, hops = self._find_candidates(src, dst, timestamp)
        if len(candidates) == 0:
            return np.empty(0, dtype=np.int64)

        trials_total = max(1, int(CONFIG.qiea.n_trials))
        agents = max(1, int(CONFIG.qiea.QIEA_agents))
        trials = max(1, trials_total // agents)
        best_per_ratio = []
        if not silent:
            print(f"QIEA: {agents} agents, {trials} trials per agent")
        if self.sparse_ratio is None:
            ratios = np.linspace(0.05, 0.95, num=19)
        else:
            ratios = [self.sparse_ratio]
        for r in ratios:
            if not silent:
                print(f"QIEA: testing sparse ratio {r}")
            max_events = max(1, int(round(float(r) * len(candidates))))
            strategy = CONFIG.qiea.func
            result,result_list,loop_list,model_time = Quantum_individual_optimizationA(
                [agents, trials], self.facade,
                (np.array([src]), np.array([dst]), np.array([timestamp])),
                candidates, 0.01, 0.01 * np.pi,
                True, True, max_events,
                float(self.facade.get_prob([src], [dst], [timestamp])[0]),
                strategy, hops
            )

            # QIEA returns diagnostic genomes as strings.  Select the best
            # genome for this ratio; build_coalitions applies the best-so-far
            # rule while preserving this increasing-ratio order.
            best_fitness = np.argmax([fitness for fitness, _count, genome in loop_list])
            best_per_ratio.append(loop_list[best_fitness])
        return best_per_ratio, candidates

    def build_coalitions(self, explanation):
        best_per_ratio, candidates = explanation 
        explanation_sort = np.argsort([count for fitness, count, genome in best_per_ratio])
        explanation_sorted = [best_per_ratio[i] for i in explanation_sort]
        coalitions_np = np.zeros((len(explanation_sorted) + 1, len(candidates)), dtype=np.int64)   
        
        best_fitness = 0.0
        for i in range(len(explanation_sorted)):
            fitness, count, genome = explanation_sorted[i]
            if fitness > best_fitness:
                best_fitness = fitness
                coalitions_np[i, :count] = np.asarray(genome.split(","), dtype=np.int64)
            else:
                coalitions_np[i, :] = coalitions_np[i-1, :]
        coalitions_np[-1, :] = candidates
        return coalitions_np, None, None
