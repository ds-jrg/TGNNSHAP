"""DyGLib adapter for TempME (Chen & Ying, NeurIPS 2023).

The upstream implementation assumes its own CSV loader and TGNN classes.  This
module keeps its motif/walk explainer while exposing the repository's standard
:class:`Explainer` interface and DyGLib ``Data``/``NeighborSampler`` objects.
"""
from __future__ import annotations

import copy
import math
import os
import pickle
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from Config.config import CONFIG
from Explainers.utils import Explainer
from DyGLib.models.modules import TGNN
from DyGLib.utils.DataLoader import Data
from DyGLib.utils.utils import BatchSubgraphs, NegativeEdgeSampler, NeighborSampler

from .models import TempME, TempME_TGAT
from .utils.batch_loader import get_item
from .utils.graph import edge_info, get_walk_finder
from .processed.data_preprocess import pre_processing, calculate_edge, marginal

from .utils.batch_loader import load_subgraph_margin

from Explainers.External.TempME.temp_exp_main import train as tempme_train


CONFIG = CONFIG()


class TempMEExplainer(Explainer):
    """Motif-based event explainer compatible with the repository evaluator."""

    def __init__(self, model: TGNN, neighbor_finder: NeighborSampler, data: Data):
        super().__init__(model, neighbor_finder, data)
        self.device = torch.device(CONFIG.model.device)
        edge_features = neighbor_finder.edge_features.detach().cpu().numpy()
        node_features = getattr(data, "node_features", None)
        if node_features is None:
            # DyGLib's current loader intentionally creates zero node features.
            node_features = model.backbone.node_features.detach().cpu().numpy()
        self.explainer =  TempME_TGAT(model, edge_features=neighbor_finder.edge_features.detach().cpu().numpy(), data=CONFIG.data.dataset_name, out_dim=CONFIG.tempME.out_dim, hid_dim=CONFIG.tempME.hid_dim, temp=CONFIG.tempME.temp,
                    n_head=1, dropout_p=CONFIG.tempME.drop_out, device=CONFIG.model.device)
        self.pack = None
        self.edge = None
        self._row_by_edge = {}

    @staticmethod
    def preprocess(data: Data, walk_finder, neg_edge_sampler: NegativeEdgeSampler):
        """Materialize motif walks in memory using DyGLib event arrays."""
        from argparse import Namespace
        args = Namespace(
            n_degree=CONFIG.model.num_neighbors)
        
        result = pre_processing(walk_finder, neg_edge_sampler, data.src_node_ids, data.dst_node_ids, data.node_interact_times, data.edge_ids)
        walks_src_new, walks_tgt_new, walks_bgd_new = marginal(result["walks_src"], result["walks_tgt"], result["walks_bgd"])
        result["walks_src_new"], result["walks_tgt_new"], result["walks_bgd_new"] = walks_src_new, walks_tgt_new, walks_bgd_new
        del result["walks_src"], result["walks_tgt"], result["walks_bgd"]
        
        edge_load = calculate_edge(result["walks_src_new"], result["walks_tgt_new"], result["walks_bgd_new"])
        pack_cat = load_subgraph_margin(args, result)
        row_by_edge = {int(edge_id): i for i, edge_id in enumerate(data.edge_ids)}
        return pack_cat, edge_load, row_by_edge

    @staticmethod
    def _preprocessing_cache_path(subset_name: str) -> str:
        if subset_name not in {"train", "test"}:
            raise ValueError("subset_name must be either 'train' or 'test'")
        return f"Data/{CONFIG.data.dataset_name}/TempME/preprocessed_{subset_name}.pkl"

    @staticmethod
    def _load_preprocessing_cache(subset_name: str):
        path = TempMEExplainer._preprocessing_cache_path(subset_name)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as file:
            return pickle.load(file)

    @staticmethod
    def _save_preprocessing_cache(subset_name: str, pack_cat, edge, row_by_edge):
        path = TempMEExplainer._preprocessing_cache_path(subset_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(
                {
                    "pack_cat": pack_cat,
                    "edge": edge,
                    "row_by_edge": row_by_edge,
                },
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @staticmethod
    def preprocess_data_if_missing(data: Data, subset_name: str = "test"):
        """Load cached preprocessing data or create and cache it if absent."""
        cached = TempMEExplainer._load_preprocessing_cache(subset_name)
        if cached is not None:
            print("Using cached TempME preprocessing data.")
            return cached["pack_cat"], cached["edge"], cached["row_by_edge"]

        print("TempME preprocessing data is missing; preprocessing walks...")
        walk_finder = get_walk_finder(data)
        neg_edge_sampler = NegativeEdgeSampler(
            data.src_node_ids,
            data.dst_node_ids,
            data.node_interact_times,
        )
        pack_cat, edge, row_by_edge = TempMEExplainer.preprocess(
            data, walk_finder, neg_edge_sampler
        )
        TempMEExplainer._save_preprocessing_cache(subset_name, pack_cat, edge, row_by_edge)
        print("Cached TempME preprocessing data.")
        return pack_cat, edge, row_by_edge

    @staticmethod
    def train_model_if_missing(model, train_neighbor_finder, full_neighbor_finder, full_random_sampler, train_data: Data, full_data: Data, device):
        """Train and cache the TempME model only when its checkpoint is absent."""
        path = f"Saved_models/{CONFIG.data.dataset_name}/TempME/{CONFIG.data.dataset_name}.pt"
        if os.path.exists(path):
            print("Using cached TempME trained model.")
            return

        train_pack_cat, train_edge, _ = TempMEExplainer.preprocess_data_if_missing(train_data, subset_name="train")
        test_pack_cat, test_edge, _ = TempMEExplainer.preprocess_data_if_missing(full_data, subset_name="test")
        
        
        from argparse import Namespace
        args = Namespace(
            base_type="tgat",
            data=CONFIG.data.dataset_name,
            out_dim=CONFIG.tempME.out_dim,
            hid_dim=CONFIG.tempME.hid_dim,
            temp=CONFIG.tempME.temp,
            n_head=1,
            device=device,
            if_bern=CONFIG.tempME.if_bern,
            drop_out=CONFIG.tempME.drop_out,
            lr=CONFIG.tempME.lr,
            weight_decay=CONFIG.tempME.weight_decay,
            n_epoch=CONFIG.tempME.n_epoch,
            bs=CONFIG.tempME.bs,
            test_bs=CONFIG.tempME.bs,
            beta=CONFIG.tempME.beta,
            prior_p=CONFIG.tempME.prior_p,
            verbose=1, # Weird name but inidicate evaluation on test set every 5 epochs
            test_threshold=True,
            save_model = True,
            ratios = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.30],
            n_degree=CONFIG.model.num_neighbors
        )
        
        print("TempME trained model is missing; training model...")
        tempme_train(args, model, train_data, full_data, train_pack_cat, test_pack_cat, train_edge, test_edge, ngh_finder=train_neighbor_finder, full_ngh_finder=full_neighbor_finder, test_rand_sampler=full_random_sampler)
        print("Cached TempME trained model.")

    def initialize(self):
        cached = TempMEExplainer._load_preprocessing_cache("test")
        if cached is None:
            raise FileNotFoundError(
                "TempME evaluation preprocessing data is missing. "
                "Call preprocess_data_if_missing(full_data, subset_name='test') first."
            )
        self.pack = cached["pack_cat"]
        self.edge = cached["edge"]
        self._row_by_edge = cached["row_by_edge"]
        print("Using cached TempME preprocessing data.")
        self.explainer.to(self.device)
        path = f"Saved_models/{CONFIG.data.dataset_name}/TempME/{CONFIG.data.dataset_name}.pt"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"TempME trained model is missing: {path}. "
                "Call train_model_if_missing() first."
            )
        self.explainer=torch.load(path, map_location=self.device, weights_only=False)
        print("Using cached TempME trained model.")

    def explain_instance(self, src: int, dst: int, timestamp: int, silent: bool = False) -> Any:
        mask = ((self.data.src_node_ids == src) & (self.data.dst_node_ids == dst) &
                (self.data.node_interact_times == timestamp))
        if not mask.any():
            raise KeyError(f"No event found for ({src}, {dst}, {timestamp})")
        edge_id = int(self.data.edge_ids[np.flatnonzero(mask)[0]])
        row = self._row_by_edge.get(edge_id)
        if row is None:
            raise KeyError(f"Event {edge_id} was not included during TempME preprocessing")
        sg_src, sg_dst, _, walks_src, walks_dst, _, _ = get_item(self.pack, np.array([row]))
        src_edge, dst_edge, _ = self.edge[:, [row]]
        with torch.no_grad():
            graphlet_imp_src = self.explainer(walks_src, np.array([src]), np.array([timestamp]), np.array([dst]))
            graphlet_imp_tgt = self.explainer(walks_dst, np.array([dst]), np.array([timestamp]), np.array([src]))
            edge_imp_src = self.explainer.retrieve_edge_imp(sg_src, graphlet_imp_src, walks_src, training=False)
            edge_imp_tgt = self.explainer.retrieve_edge_imp(sg_dst, graphlet_imp_tgt, walks_dst, training=False)
        def make_batch_subgraph(subgraph, neighbor_finder, device):
            """Convert a precomputed subgraph to the container expected by TGAT."""
            edge_features = neighbor_finder.get_edge_features_for_multi_hop(subgraph[1])
            batch_subgraph = BatchSubgraphs(*subgraph, event_features=edge_features)
            batch_subgraph.to(device)
            return batch_subgraph
        src_sg = make_batch_subgraph(sg_src, self.neighbor_finder, CONFIG.model.device)
        dst_sg = make_batch_subgraph(sg_dst, self.neighbor_finder, CONFIG.model.device)
            
            
        scores = {}
        for events, importance in ((src_sg.events, edge_imp_src), (dst_sg.events, edge_imp_tgt)):
            for i in range(len(events)):
                for e, imp in zip(events[i].reshape(-1), importance[i].reshape(-1)):
                    scores[int(e)] = max(scores.get(int(e), 0.0), float(imp))
        return scores, src_sg, dst_sg

    def build_coalitions(self, explanation: Tuple[dict, BatchSubgraphs, BatchSubgraphs]):
        scores, sg_src, sg_dst = explanation
        ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranking = [edge for edge, _ in ranking]
        coalitions = np.zeros((len(ranking), len(ranking)), dtype=np.int64)
        for i in range(len(ranking)):
            coalitions[i, :i + 1] = ranking[:i + 1]
        return coalitions, sg_src, sg_dst
