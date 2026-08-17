"""DyGLib adapter for TempME (Chen & Ying, NeurIPS 2023).

The upstream implementation assumes its own CSV loader and TGNN classes.  This
module keeps its motif/walk explainer while exposing the repository's standard
:class:`Explainer` interface and DyGLib ``Data``/``NeighborSampler`` objects.
"""
from __future__ import annotations

import copy
import math
import os
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

    def preprocess(self, walk_finder, neg_edge_sampler: NegativeEdgeSampler, train: bool = True):
        """Materialize motif walks in memory using DyGLib event arrays."""
        mask = ~np.isnan(self.data.labels) if np.issubdtype(self.data.labels.dtype, np.floating) else np.ones(len(self.data.edge_ids), dtype=bool)
        src = self.data.src_node_ids[mask]
        dst = self.data.dst_node_ids[mask]
        ts = self.data.node_interact_times[mask]
        edges = self.data.edge_ids[mask]
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

        self.pack = self._concat_pack(batches)
        self.edge = np.stack([edge_info(self.pack[i][1]) for i in (3, 4, 5)], axis=0)
        self._row_by_edge = {int(edge_id): i for i, edge_id in enumerate(edges)}

    @staticmethod
    def _concat_pack(batches):
        def merge_subgraphs(parts):
            return tuple([
                [np.concatenate([p[j][layer] for p in parts], axis=0)
                 for layer in range(len(parts[0][j]))]
                for j in range(3)
            ])
        def merge_walks(parts):
            merged = [np.concatenate([p[j] for p in parts], axis=0) for j in range(3)]
            # The original preprocessing appends categorical motif and
            # marginal features.  DyGLib does not persist them, so use the
            # neutral category/marginal for the adapter's five-field format.
            shape = merged[0].shape[:2] + (1,)
            merged.extend([np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=np.float32)])
            return tuple(merged)
        return (merge_subgraphs(batches["subgraph_src"]), merge_subgraphs(batches["subgraph_tgt"]),
                merge_subgraphs(batches["subgraph_bgd"]), merge_walks(batches["walks_src"]),
                merge_walks(batches["walks_tgt"]), merge_walks(batches["walks_bgd"]),
                np.concatenate(batches["dst_fake"], axis=0))

    def initialize(self, train: bool = False):
        if self.pack is None:
            raise RuntimeError("TempME preprocessing is required before initialize()")
        self.explainer.to(self.device)
        path = f"Saved_models/{CONFIG.data.dataset_name}/TempMe/Explainer.pt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if train:
            self.train()
        elif not os.path.exists(path):
            raise FileNotFoundError(f"TempME checkpoint not found: {path}. Run with --preprocessing true.")
        else:
            self.explainer.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))

    def train(self):
        """Train the motif selector while keeping the DyGLib predictor frozen."""
        optimizer = torch.optim.Adam(self.explainer.parameters(), lr=CONFIG.tempME.lr, weight_decay=CONFIG.tempME.weight_decay)
        criterion = nn.MSELoss() if CONFIG.model.task == "regression" else nn.BCEWithLogitsLoss()
        src, dst, ts, edges = self.data.src_node_ids, self.data.dst_node_ids, self.data.node_interact_times, self.data.edge_ids
        n = min(len(src), self.pack[3][0].shape[0])
        for epoch in range(CONFIG.tempME.n_epoch):
            losses = []
            for start in range(0, n, CONFIG.tempME.bs):
                idx = np.arange(start, min(start + CONFIG.tempME.bs, n))
                sg_src, sg_dst, _, walks_src, walks_dst, _, _ = get_item(self.pack, idx)
                src_edge, dst_edge, _ = self.edge[:, idx]
                bsz = len(idx)
                def make_sg(sg):
                    feats = self.neighbor_finder.get_edge_features_for_multi_hop(sg[1])
                    result = BatchSubgraphs(*sg, feats)
                    result.to(self.device)
                    return result
                src_sg, dst_sg = make_sg(sg_src), make_sg(sg_dst)
                with torch.no_grad():
                    target = self.model(src[idx], dst[idx], ts[idx], src_sg, dst_sg, edges_are_positive=False).detach()
                p_src = self.explainer(walks_src, ts[idx], src_edge)
                p_dst = self.explainer(walks_dst, ts[idx], dst_edge)
                explanation = self.explainer.retrieve_explanation(sg_src, p_src, walks_src, sg_dst, p_dst, walks_dst, sg_dst, p_dst, walks_dst, training=CONFIG.tempME.if_bern)
                src_sg.set_event_attention([x[:bsz] for x in explanation])
                dst_sg.set_event_attention([x[bsz:2 * bsz] for x in explanation])
                pred = self.model(src[idx], dst[idx], ts[idx], src_sg, dst_sg, edges_are_positive=False)
                pred_loss = torch.nan_to_num(criterion(pred, target), nan=0.0, posinf=1e6, neginf=1e6)
                kl = self.explainer.kl_loss(p_src, walks_src, target=CONFIG.tempME.prior_p) + self.explainer.kl_loss(p_dst, walks_dst, target=CONFIG.tempME.prior_p)
                loss = torch.nan_to_num(pred_loss + CONFIG.tempME.beta * kl, nan=0.0, posinf=1e6, neginf=1e6)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(loss.item())
            print(f"TempME epoch {epoch}: {np.mean(losses):.6f}")
        torch.save(self.explainer.state_dict(), f"Saved_models/{CONFIG.data.dataset_name}/TempMe/Explainer.pt")

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
