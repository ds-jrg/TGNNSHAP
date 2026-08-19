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

from .models import TempME
from .utils.batch_loader import get_item
from .utils.graph import edge_info, get_walk_finder
from .processed.data_preprocess import marginal, calculate_edge


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
        self.explainer = TempME(
            # DyGLib always evaluates two sampled hops, so TempME must return
            # one attention tensor for each hop (the upstream ``tgn`` path).
            model, base_model_type="tgn",
            data=CONFIG.data.dataset_name, out_dim=CONFIG.tempME.out_dim,
            hid_dim=CONFIG.tempME.hid_dim, temp=CONFIG.tempME.temp,
            if_cat_feature=True, dropout_p=CONFIG.tempME.drop_out,
            device=self.device, edge_raw_features=edge_features,
            node_raw_features=node_features,
        )
        self.pack = None
        self.edge = None
        self._row_by_edge = {}

    @staticmethod
    def preprocess(data: Data, walk_finder, neg_edge_sampler: NegativeEdgeSampler):
        """Materialize motif walks in memory using DyGLib event arrays."""
        mask = ~np.isnan(data.labels) if np.issubdtype(data.labels.dtype, np.floating) else np.ones(len(data.edge_ids), dtype=bool)
        src = data.src_node_ids[mask]
        dst = data.dst_node_ids[mask]
        ts = data.node_interact_times[mask]
        edges = data.edge_ids[mask]
        batches = {name: [] for name in ("subgraph_src", "subgraph_tgt", "subgraph_bgd", "walks_src", "walks_tgt", "walks_bgd", "dst_fake")}
        degree = CONFIG.model.num_neighbors
        for start in tqdm(range(0, len(src), max(1, CONFIG.tempME.bs)), desc="TempME walks"):
            sl = slice(start, min(start + CONFIG.tempME.bs, len(src)))
            s, d, t, e = src[sl], dst[sl], ts[sl], edges[sl]
            _, fake = neg_edge_sampler.sample(len(s))
            subgraphs = [
                walk_finder.find_k_hop(2, s, t, degree, e_idx_l=e),
                walk_finder.find_k_hop(2, d, t, degree, e_idx_l=e),
                walk_finder.find_k_hop(2, fake, t, degree),
            ]
            walks = [walk_finder.find_k_walks(degree, root, 3, sg) for root, sg in zip((s, d, fake), subgraphs)]
            for name, sg in zip(("subgraph_src", "subgraph_tgt", "subgraph_bgd"), subgraphs):
                batches[name].append(sg)
            for name, walk in zip(("walks_src", "walks_tgt", "walks_bgd"), walks):
                n, ei, ti, anon = walk
                batches[name].append((n.astype(np.int64), ei.astype(np.int64), ti.astype(np.float32), anon.astype(np.int64)))
            batches["dst_fake"].append(fake)

        pack = TempMEExplainer._concat_pack(batches)
        new_walks = marginal(pack[3], pack[4], pack[5])
        pack = (pack[0], pack[1], pack[2], new_walks[0], new_walks[1], new_walks[2], pack[6])
        edge = calculate_edge(pack[3], pack[4], pack[5])
        row_by_edge = {int(edge_id): i for i, edge_id in enumerate(edges)}
        return pack, edge, row_by_edge

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
    def _save_preprocessing_cache(subset_name: str, pack, edge, row_by_edge):
        path = TempMEExplainer._preprocessing_cache_path(subset_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(
                {
                    "pack": pack,
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
            return cached["pack"], cached["edge"], cached["row_by_edge"]

        print("TempME preprocessing data is missing; preprocessing walks...")
        walk_finder = get_walk_finder(data)
        neg_edge_sampler = NegativeEdgeSampler(
            data.src_node_ids,
            data.dst_node_ids,
            data.node_interact_times,
        )
        pack, edge, row_by_edge = TempMEExplainer.preprocess(
            data, walk_finder, neg_edge_sampler
        )
        TempMEExplainer._save_preprocessing_cache(subset_name, pack, edge, row_by_edge)
        print("Cached TempME preprocessing data.")
        return pack, edge, row_by_edge
    
    @staticmethod
    def train_model_if_missing(model, neighbor_finder, data: Data, device):
        """Train and cache the TempME model only when its checkpoint is absent."""
        path = f"Saved_models/{CONFIG.data.dataset_name}/TempME/Explainer.pt"
        if os.path.exists(path):
            print("Using cached TempME trained model.")
            return

        pack, edge, _ = TempMEExplainer.preprocess_data_if_missing(data, subset_name="train")
        edge_features = neighbor_finder.edge_features.detach().cpu().numpy()
        node_features = getattr(data, "node_features", None)
        if node_features is None:
            node_features = model.backbone.node_features.detach().cpu().numpy()
        explainer = TempME(
            model, base_model_type="tgn", data=CONFIG.data.dataset_name,
            out_dim=CONFIG.tempME.out_dim, hid_dim=CONFIG.tempME.hid_dim,
            temp=CONFIG.tempME.temp, if_cat_feature=True,
            dropout_p=CONFIG.tempME.drop_out, device=device,
            edge_raw_features=edge_features, node_raw_features=node_features,
        )
        explainer.to(device)
        print("TempME trained model is missing; training model...")
        TempMEExplainer.train(model, explainer, neighbor_finder, data, pack, edge, device)
        print("Cached TempME trained model.")

    @staticmethod
    def _concat_pack(batches):
        def merge_subgraphs(parts):
            return tuple([
                [np.concatenate([p[j][layer] for p in parts], axis=0)
                 for layer in range(len(parts[0][j]))]
                for j in range(3)
            ])
        def merge_walks(parts):
            merged = [np.concatenate([p[j] for p in parts], axis=0) for j in range(4)]
            return np.concatenate(merged, axis=2)
        return (merge_subgraphs(batches["subgraph_src"]), merge_subgraphs(batches["subgraph_tgt"]),
                merge_subgraphs(batches["subgraph_bgd"]), merge_walks(batches["walks_src"]),
                merge_walks(batches["walks_tgt"]), merge_walks(batches["walks_bgd"]),
                np.concatenate(batches["dst_fake"], axis=0))

    def initialize(self):
        cached = TempMEExplainer._load_preprocessing_cache("test")
        if cached is None:
            raise FileNotFoundError(
                "TempME evaluation preprocessing data is missing. "
                "Call preprocess_data_if_missing(full_data, subset_name='test') first."
            )
        self.pack = cached["pack"]
        self.edge = cached["edge"]
        self._row_by_edge = cached["row_by_edge"]
        print("Using cached TempME preprocessing data.")
        self.explainer.to(self.device)
        path = f"Saved_models/{CONFIG.data.dataset_name}/TempME/Explainer.pt"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"TempME trained model is missing: {path}. "
                "Call train_model_if_missing() first."
            )
        self.explainer.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        print("Using cached TempME trained model.")

    @staticmethod
    def train(model, explainer, neighbor_finder, data, pack, edge, device):
        """Train the motif selector while keeping the DyGLib predictor frozen."""
        optimizer = torch.optim.Adam(explainer.parameters(), lr=CONFIG.tempME.lr, weight_decay=CONFIG.tempME.weight_decay)
        criterion = nn.MSELoss() if CONFIG.model.task == "regression" else nn.BCEWithLogitsLoss()
        src, dst, ts, edges = data.src_node_ids, data.dst_node_ids, data.node_interact_times, data.edge_ids
        n = min(len(src), pack[3][0].shape[0])
        for epoch in range(CONFIG.tempME.n_epoch):
            losses = []
            for start in range(0, n, CONFIG.tempME.bs):
                idx = np.arange(start, min(start + CONFIG.tempME.bs, n))
                sg_src, sg_dst, _, walks_src, walks_dst, _, _ = get_item(pack, idx)
                src_edge, dst_edge, _ = edge[:, idx]
                bsz = len(idx)
                def make_sg(sg):
                    feats = neighbor_finder.get_edge_features_for_multi_hop(sg[1])
                    result = BatchSubgraphs(*sg, feats)
                    result.to(device)
                    return result
                src_sg, dst_sg = make_sg(sg_src), make_sg(sg_dst)
                with torch.no_grad():
                    target = model(src[idx], dst[idx], ts[idx], src_sg, dst_sg, edges_are_positive=False).detach()
                p_src = explainer(walks_src, ts[idx], src_edge)
                p_dst = explainer(walks_dst, ts[idx], dst_edge)
                explanation = explainer.retrieve_explanation(sg_src, p_src, walks_src, sg_dst, p_dst, walks_dst, sg_dst, p_dst, walks_dst, training=CONFIG.tempME.if_bern)
                src_sg.set_event_attention([x[:bsz] for x in explanation])
                dst_sg.set_event_attention([x[bsz:2 * bsz] for x in explanation])
                pred = model(src[idx], dst[idx], ts[idx], src_sg, dst_sg, edges_are_positive=False)
                pred_loss = torch.nan_to_num(criterion(pred, target), nan=0.0, posinf=1e6, neginf=1e6)
                kl = explainer.kl_loss(p_src, walks_src, target=CONFIG.tempME.prior_p) + explainer.kl_loss(p_dst, walks_dst, target=CONFIG.tempME.prior_p)
                loss = torch.nan_to_num(pred_loss + CONFIG.tempME.beta * kl, nan=0.0, posinf=1e6, neginf=1e6)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(loss.item())
            print(f"TempME epoch {epoch}: {np.mean(losses):.6f}")
        os.mkdir(f"Saved_models/{CONFIG.data.dataset_name}/TempME") if not os.path.exists(f"Saved_models/{CONFIG.data.dataset_name}/TempME") else None
        torch.save(explainer.state_dict(), f"Saved_models/{CONFIG.data.dataset_name}/TempME/Explainer.pt")

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
            p_src = self.explainer(walks_src, np.array([timestamp]), src_edge).squeeze(-1).cpu().numpy()[0]
            p_dst = self.explainer(walks_dst, np.array([timestamp]), dst_edge).squeeze(-1).cpu().numpy()[0]
        scores = {}
        for walk, probs in ((walks_src, p_src), (walks_dst, p_dst)):
            for event_id, score in zip(walk[1][0].reshape(-1), np.repeat(probs, 3)):
                if event_id:
                    scores[int(event_id)] = max(scores.get(int(event_id), 0.0), float(score))
        sg_src_b = BatchSubgraphs(*sg_src, self.neighbor_finder.get_edge_features_for_multi_hop(sg_src[1]))
        sg_dst_b = BatchSubgraphs(*sg_dst, self.neighbor_finder.get_edge_features_for_multi_hop(sg_dst[1]))
        events = np.unique(np.concatenate([sg_src_b.get_events(), sg_dst_b.get_events()], axis=1))
        events = events[events != 0]
        ranked = sorted((int(e) for e in events), key=lambda e: (-scores.get(e, 0.0), e))
        return np.asarray(ranked, dtype=np.int64), scores, sg_src_b, sg_dst_b

    def build_coalitions(self, explanation: Tuple[np.ndarray, dict, BatchSubgraphs, BatchSubgraphs]):
        ranked, scores, sg_src, sg_dst = explanation
        if len(ranked) == 0:
            return np.zeros((1, 1), dtype=np.int64), sg_src, sg_dst
        coalitions = np.zeros((len(ranked), len(ranked)), dtype=np.int64)
        for i in range(len(ranked)):
            coalitions[i, :i + 1] = ranked[:i + 1]
        return coalitions, sg_src, sg_dst
