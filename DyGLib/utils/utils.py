"""
Utilities for training dynamic-graph (temporal graph) models.

This module bundles together:
    * generic helpers (seeding, moving tensors to GPU, building optimizers, ...)
    * a `NeighborSampler` for pulling temporal neighborhoods out of an
      interaction graph (uniform / recent / time-interval-aware strategies)
    * a `NegativeEdgeSampler` for generating negative edges under the
      "random" / "historical" / "inductive" evaluation protocols
    * `BatchSubgraphs`, a container that stores a batch of multi-hop temporal
      subgraphs and offers a range of masking / slicing / merging operations
      used during batching and message-passing.
"""

from typing import Optional, List, Tuple
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from Config.config import CONFIG
from DyGLib.utils.DataLoader import Data

CONFIG = CONFIG()


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #

def set_random_seed(seed: int = 0):
    """
    Seed every RNG (python, numpy, torch, cudnn) for reproducibility.

    :param seed: int, random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Force deterministic (but slower) cuDNN kernels.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_to_gpu(*data, device: str):
    """
    Move one or more objects (tensors, modules, ...) onto `device`.

    :param data: any number of objects exposing a `.to(device)` method
    :param device: str, target device, e.g. 'cuda:0' or 'cpu'
    :return: the moved object if a single item was passed, otherwise a tuple
    """
    moved = []
    for item in data:
        item = item.to(device)
        moved.append(item)

    if len(moved) > 1:
        return tuple(moved)
    return moved[0]


def get_parameter_sizes(model: nn.Module):
    """
    Count the number of trainable parameters in a model.

    :param model: nn.Module
    :return: int, total number of trainable scalar parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_optimizer(model: nn.Module, optimizer_name: str, learning_rate: float, weight_decay: float = 0.0):
    """
    Build a torch optimizer for `model`.

    :param model: nn.Module
    :param optimizer_name: str, one of 'Adam', 'SGD', 'RMSprop'
    :param learning_rate: float
    :param weight_decay: float, L2 regularization coefficient
    :return: torch.optim.Optimizer
    """
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Wrong value for optimizer {optimizer_name}!")

    return optimizer


# --------------------------------------------------------------------------- #
# Temporal neighbor sampling
# --------------------------------------------------------------------------- #

class NeighborSampler:
    """
    Samples temporal neighborhoods for a set of query nodes.

    For every node, all of its interactions (neighbor id, edge id, timestamp)
    are pre-sorted by time so that "give me everything before time t" can be
    answered with a binary search instead of a linear scan.
    """

    def __init__(self, adj_list: list, edge_features: np.ndarray, edge_labels: Optional[np.ndarray] = None,
                 sample_neighbor_strategy: str = 'uniform', time_scaling_factor: float = 0.0, seed=None):
        """
        :param adj_list: list of list, where adj_list[node_id] is a list of
            (neighbor_id, edge_id, timestamp) triples describing every
            interaction that touches `node_id`
        :param edge_features: ndarray, per-edge feature matrix
        :param edge_labels: optional ndarray, per-edge labels
        :param sample_neighbor_strategy: str, one of 'uniform', 'recent', 'time_interval_aware'
        :param time_scaling_factor: float, only used when sample_neighbor_strategy == 'time_interval_aware';
            larger values bias sampling more strongly towards recent interactions
        :param seed: optional int, random seed for reproducible sampling
        """
        self.sample_neighbor_strategy = sample_neighbor_strategy
        self.seed = seed

        # Per-node, time-sorted lists of neighbor ids / edge ids / interaction times.
        self.nodes_neighbor_ids = []
        self.nodes_edge_ids = []
        self.nodes_neighbor_times = []

        self.edge_features = torch.from_numpy(edge_features)

        if edge_labels is not None:
            self.edge_labels = edge_labels

        if self.sample_neighbor_strategy == 'time_interval_aware':
            self.nodes_neighbor_sampled_probabilities = []
            self.time_scaling_factor = time_scaling_factor

        # `adj_list[0]` is always empty by convention (node ids start at 1),
        # so the corresponding entries below simply end up being empty arrays.
        for node_idx, per_node_neighbors in enumerate(adj_list):
            # Sort each node's interactions by timestamp. `sorted()` is stable,
            # and sorting by edge id would be equally valid since interactions
            # are chronological in the original data file.
            sorted_per_node_neighbors = sorted(per_node_neighbors, key=lambda x: x[2])
            self.nodes_neighbor_ids.append(np.array([x[0] for x in sorted_per_node_neighbors]))
            self.nodes_edge_ids.append(np.array([x[1] for x in sorted_per_node_neighbors]))
            self.nodes_neighbor_times.append(np.array([x[2] for x in sorted_per_node_neighbors]))

            # Precompute sampling probabilities for the time-interval-aware
            # strategy proposed in the CAWN paper.
            if self.sample_neighbor_strategy == 'time_interval_aware':
                self.nodes_neighbor_sampled_probabilities.append(
                    self.compute_sampled_probabilities(np.array([x[2] for x in sorted_per_node_neighbors]))
                )

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def get_edge_features(self, edge_ids: np.ndarray):
        """Look up feature vectors for a batch of edge ids."""
        return self.edge_features[edge_ids]

    def get_edge_labels(self, edge_ids: np.ndarray):
        """Look up labels for a batch of edge ids."""
        return self.edge_labels[edge_ids]

    def get_edge_labels_for_multi_hop(self, edge_ids: List[np.ndarray]):
        """Apply `get_edge_labels` to each hop's edge-id array."""
        return [self.get_edge_labels(x) for x in edge_ids]

    def get_edge_features_for_multi_hop(self, edge_ids: List[np.ndarray]):
        """Apply `get_edge_features` to each hop's edge-id array."""
        return [self.get_edge_features(x) for x in edge_ids]

    def compute_sampled_probabilities(self, node_neighbor_times: np.ndarray):
        """
        Compute sampling probabilities for a node's historical neighbors,
        weighted so that more recent interactions are more likely to be
        picked (time-interval-aware strategy).

        :param node_neighbor_times: ndarray, shape (num_historical_neighbors, )
        :return: ndarray, shape (num_historical_neighbors, )
        """
        if len(node_neighbor_times) == 0:
            return np.array([])

        # Express times as deltas relative to the most recent interaction.
        node_neighbor_times = node_neighbor_times - np.max(node_neighbor_times)
        exp_node_neighbor_times = np.exp(self.time_scaling_factor * node_neighbor_times)
        sampled_probabilities = exp_node_neighbor_times / np.cumsum(exp_node_neighbor_times)

        # The earliest entries in exp_node_neighbor_times can be ~0, which
        # makes the corresponding probability `nan` (0 / 0). We flag those
        # with a very large negative number so a later softmax pushes their
        # probability towards 0 instead of crashing.
        sampled_probabilities[np.isnan(sampled_probabilities)] = -1e10
        return sampled_probabilities

    def find_neighbors_before(self, node_id: int, interact_time: float, kept_edge_ids: Optional[np.ndarray] = None,
                               return_sampled_probabilities: bool = False):
        """
        Return every interaction of `node_id` that happened strictly before
        `interact_time`, sorted by time (optionally filtered down to a set
        of `kept_edge_ids`).

        :param node_id: int
        :param interact_time: float
        :param kept_edge_ids: optional ndarray, restrict results to these edge ids
        :param return_sampled_probabilities: bool, also return the precomputed
            time-interval-aware sampling probabilities
        :return: (neighbor_ids, edge_ids, timestamps, sampled_probabilities_or_None),
            each of shape (num_historical_neighbors, )
        """
        node_id = int(node_id)  # in case it's a numpy int
        # searchsorted returns index i such that list[i - 1] < interact_time <= list[i].
        i = np.searchsorted(self.nodes_neighbor_times[node_id], interact_time)

        if kept_edge_ids is not None:
            mask = np.isin(self.nodes_edge_ids[node_id][:i], kept_edge_ids)
        else:
            mask = np.full_like(self.nodes_edge_ids[node_id][:i], True, dtype="bool")

        if return_sampled_probabilities:
            return (self.nodes_neighbor_ids[node_id][:i][mask],
                    self.nodes_edge_ids[node_id][:i][mask],
                    self.nodes_neighbor_times[node_id][:i][mask],
                    self.nodes_neighbor_sampled_probabilities[node_id][:i][mask])

        return (self.nodes_neighbor_ids[node_id][:i][mask],
                self.nodes_edge_ids[node_id][:i][mask],
                self.nodes_neighbor_times[node_id][:i][mask],
                None)

    def get_historical_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20,
                                  kept_edge_ids: Optional[np.ndarray] = None):
        """
        For each (node_id, node_interact_time) pair, sample a fixed-size
        window of `num_neighbors` historical neighbors.

        :param node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param num_neighbors: int, number of neighbors to sample per node
        :param kept_edge_ids: optional ndarray, per-node allow-lists of edge ids
        :return: (nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times),
            each ndarray of shape (batch_size, num_neighbors)
        """
        assert num_neighbors > 0, 'Number of sampled neighbors for each node should be greater than 0!'

        # Entry (i, j) = id of the j-th sampled neighbor of node_ids[i],
        # restricted to interactions before node_interact_times[i].
        nodes_neighbor_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
        # Entry (i, j) = id of the edge connecting node_ids[i] to nodes_neighbor_ids[i, j].
        nodes_edge_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
        # Entry (i, j) = timestamp of that interaction.
        nodes_neighbor_times = np.zeros((len(node_ids), num_neighbors)).astype(np.float32)

        for idx, (node_id, node_interact_time) in enumerate(zip(node_ids, node_interact_times)):
            kept_edges = kept_edge_ids[idx] if kept_edge_ids is not None else None
            node_neighbor_ids, node_edge_ids, node_neighbor_times, node_neighbor_sampled_probabilities = \
                self.find_neighbors_before(
                    node_id=node_id,
                    interact_time=node_interact_time,
                    kept_edge_ids=kept_edges,
                    return_sampled_probabilities=self.sample_neighbor_strategy == 'time_interval_aware',
                )

            if len(node_neighbor_ids) == 0:
                continue

            if self.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
                # 'uniform'             -> shuffle before sampling (probabilities = None)
                # 'time_interval_aware' -> sample according to the precomputed probabilities
                if node_neighbor_sampled_probabilities is not None:
                    # torch.softmax handles the degenerate case where
                    # node_neighbor_sampled_probabilities is entirely -1e10.
                    node_neighbor_sampled_probabilities = torch.softmax(
                        torch.from_numpy(node_neighbor_sampled_probabilities).float(), dim=0
                    ).numpy()

                if self.seed is None:
                    sampled_indices = np.random.choice(a=len(node_neighbor_ids), size=num_neighbors,
                                                        p=node_neighbor_sampled_probabilities)
                else:
                    sampled_indices = self.random_state.choice(a=len(node_neighbor_ids), size=num_neighbors,
                                                                 p=node_neighbor_sampled_probabilities)

                nodes_neighbor_ids[idx, :] = node_neighbor_ids[sampled_indices]
                nodes_edge_ids[idx, :] = node_edge_ids[sampled_indices]
                nodes_neighbor_times[idx, :] = node_neighbor_times[sampled_indices]

                # Re-sort the sampled window by timestamp (ascending). This is
                # not strictly required by TGAT / CAWN, since both compute in
                # an order-agnostic / per-walk manner, but we keep it for
                # consistency and easier debugging.
                sorted_position = nodes_neighbor_times[idx, :].argsort()
                nodes_neighbor_ids[idx, :] = nodes_neighbor_ids[idx, :][sorted_position]
                nodes_edge_ids[idx, :] = nodes_edge_ids[idx, :][sorted_position]
                nodes_neighbor_times[idx, :] = nodes_neighbor_times[idx, :][sorted_position]

            elif self.sample_neighbor_strategy == 'recent':
                # Keep only the `num_neighbors` most recent interactions,
                # right-aligned (older/missing entries stay at 0 on the left).
                node_neighbor_ids = node_neighbor_ids[-num_neighbors:]
                node_edge_ids = node_edge_ids[-num_neighbors:]
                node_neighbor_times = node_neighbor_times[-num_neighbors:]

                nodes_neighbor_ids[idx, num_neighbors - len(node_neighbor_ids):] = node_neighbor_ids
                nodes_edge_ids[idx, num_neighbors - len(node_edge_ids):] = node_edge_ids
                nodes_neighbor_times[idx, num_neighbors - len(node_neighbor_times):] = node_neighbor_times

            else:
                raise ValueError(f'Not implemented error for sample_neighbor_strategy {self.sample_neighbor_strategy}!')

        return nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times

    def get_multi_hop_neighbors(self, num_hops: int, node_ids: np.ndarray, node_interact_times: np.ndarray,
                                 num_neighbors: int = 20, kept_edge_ids: Optional[np.ndarray] = None):
        """
        Expand `get_historical_neighbors` outward for `num_hops` hops,
        treating each hop's sampled neighbors as the query nodes for the
        next hop.

        :param num_hops: int, number of hops to expand
        :param node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param num_neighbors: int, neighbors sampled per node at each hop
        :param kept_edge_ids: optional ndarray, per-node allow-lists of edge ids
        :return: (nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list),
            each a list of `num_hops` ndarrays; the h-th ndarray has shape
            (batch_size, num_neighbors ** (h + 1))
        """
        assert num_hops > 0, 'Number of sampled hops should be greater than 0!'

        # First hop.
        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_historical_neighbors(
            node_ids=node_ids, node_interact_times=node_interact_times,
            num_neighbors=num_neighbors, kept_edge_ids=kept_edge_ids,
        )

        nodes_neighbor_ids_list = [nodes_neighbor_ids]
        nodes_edge_ids_list = [nodes_edge_ids]
        nodes_neighbor_times_list = [nodes_neighbor_times]

        # Every neighbor sampled at hop h inherits the same allow-list as its parent.
        kept_edge_ids = kept_edge_ids.repeat(num_neighbors, axis=0) if kept_edge_ids is not None else None

        for hop in range(1, num_hops):
            # Treat the previous hop's sampled neighbors as the new query nodes.
            nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_historical_neighbors(
                node_ids=nodes_neighbor_ids_list[-1].flatten(),
                node_interact_times=nodes_neighbor_times_list[-1].flatten(),
                num_neighbors=num_neighbors, kept_edge_ids=kept_edge_ids,
            )

            # Reshape back to (batch_size, num_neighbors ** (hop + 1)).
            nodes_neighbor_ids = nodes_neighbor_ids.reshape(len(node_ids), -1)
            nodes_edge_ids = nodes_edge_ids.reshape(len(node_ids), -1)
            nodes_neighbor_times = nodes_neighbor_times.reshape(len(node_ids), -1)

            nodes_neighbor_ids_list.append(nodes_neighbor_ids)
            nodes_edge_ids_list.append(nodes_edge_ids)
            nodes_neighbor_times_list.append(nodes_neighbor_times)

        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def get_all_first_hop_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray,
                                     kept_edge_ids: Optional[np.ndarray] = None):
        """
        Return *all* first-hop historical neighbors (no fixed-size sampling),
        one ragged list per node.

        :param node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param kept_edge_ids: optional ndarray, per-node allow-lists of edge ids
        :return: (nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list),
            each a python list of length batch_size containing variable-length ndarrays
        """
        nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list = [], [], []

        for idx, (node_id, node_interact_time) in enumerate(zip(node_ids, node_interact_times)):
            kept_edges = kept_edge_ids[idx] if kept_edge_ids is not None else None
            node_neighbor_ids, node_edge_ids, node_neighbor_times, _ = self.find_neighbors_before(
                node_id=node_id, interact_time=node_interact_time,
                return_sampled_probabilities=False, kept_edge_ids=kept_edges,
            )
            nodes_neighbor_ids_list.append(node_neighbor_ids)
            nodes_edge_ids_list.append(node_edge_ids)
            nodes_neighbor_times_list.append(node_neighbor_times)

        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def reset_random_state(self):
        """Reinitialize the RNG from `self.seed` (useful before eval passes)."""
        self.random_state = np.random.RandomState(self.seed)


def get_neighbor_sampler(data: Data, edge_features: np.ndarray, sample_neighbor_strategy: str = 'uniform',
                          time_scaling_factor: float = 0.0, seed: Optional[int] = None):
    """
    Build a `NeighborSampler` from a `Data` object by first constructing the
    adjacency list of (neighbor, edge, timestamp) triples.

    :param data: Data
    :param edge_features: ndarray, per-edge feature matrix
    :param sample_neighbor_strategy: str, 'uniform', 'recent', or 'time_interval_aware'
    :param time_scaling_factor: float, see `NeighborSampler.__init__`
    :param seed: optional int, random seed
    :return: NeighborSampler
    """
    max_node_id = max(data.src_node_ids.max(), data.dst_node_ids.max())

    # adj_list[0] is intentionally left empty.
    adj_list = [[] for _ in range(max_node_id + 1)]
    for src_node_id, dst_node_id, edge_id, node_interact_time in zip(
            data.src_node_ids, data.dst_node_ids, data.edge_ids, data.node_interact_times):
        if not CONFIG.data.is_directed:
            adj_list[src_node_id].append((dst_node_id, edge_id, node_interact_time))
        adj_list[dst_node_id].append((src_node_id, edge_id, node_interact_time))

    return NeighborSampler(adj_list=adj_list, edge_features=edge_features, edge_labels=data.types,
                            sample_neighbor_strategy=sample_neighbor_strategy,
                            time_scaling_factor=time_scaling_factor, seed=seed)


# --------------------------------------------------------------------------- #
# Negative edge sampling
# --------------------------------------------------------------------------- #

class NegativeEdgeSampler(object):
    """
    Generates negative (non-observed) edges for link-prediction training and
    evaluation, supporting three strategies:

        * 'random'     - source/destination drawn independently and uniformly
        * 'historical' - prefer edges seen earlier in history but absent from
                          the current batch
        * 'inductive'  - like 'historical', but additionally excludes edges
                          seen during the "observed" (e.g. training) period
    """

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, interact_times: np.ndarray,
                 last_observed_time: Optional[float] = None, negative_sample_strategy: str = 'random',
                 seed: Optional[int] = None):
        """
        :param src_node_ids: ndarray, shape (num_edges, ), source node ids
        :param dst_node_ids: ndarray, shape (num_edges, ), destination node ids
        :param interact_times: ndarray, shape (num_edges, ), interaction timestamps
        :param last_observed_time: optional float, end of the "observed" period (needed for 'inductive')
        :param negative_sample_strategy: str, 'random', 'historical', or 'inductive'
        :param seed: optional int, random seed
        """
        self.seed = seed
        self.negative_sample_strategy = negative_sample_strategy
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.interact_times = interact_times
        self.unique_src_node_ids = np.unique(src_node_ids)
        self.unique_dst_node_ids = np.unique(dst_node_ids)
        self.unique_interact_times = np.unique(interact_times)
        self.earliest_time = min(self.unique_interact_times)
        self.last_observed_time = last_observed_time

        if self.negative_sample_strategy != 'random':
            # Full cartesian product of observed source/destination ids -
            # the universe of edges we're allowed to sample negatives from.
            self.possible_edges = set(
                (src_node_id, dst_node_id)
                for src_node_id in self.unique_src_node_ids
                for dst_node_id in self.unique_dst_node_ids
            )

        if self.negative_sample_strategy == 'inductive':
            self.observed_edges = self.get_unique_edges_between_start_end_time(self.earliest_time, self.last_observed_time or self.earliest_time)

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def get_unique_edges_between_start_end_time(self, start_time: float, end_time: float):
        """
        Collect the set of unique (src, dst) edges observed in [start_time, end_time].

        :param start_time: float
        :param end_time: float
        :return: set of (src_node_id, dst_node_id) tuples
        """
        selected_time_interval = np.logical_and(self.interact_times >= start_time, self.interact_times <= end_time)
        return set(
            (src_node_id, dst_node_id)
            for src_node_id, dst_node_id in zip(self.src_node_ids[selected_time_interval],
                                                 self.dst_node_ids[selected_time_interval])
        )

    def sample(self, size: int, batch_src_node_ids: Optional[np.ndarray] = None, batch_dst_node_ids: Optional[np.ndarray] = None,
               current_batch_start_time: float = 0.0, current_batch_end_time: float = 0.0):
        """
        Dispatch to the configured negative sampling strategy.

        :param size: int, number of negative edges to sample
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination ids in the current batch
        :param current_batch_start_time: float
        :param current_batch_end_time: float
        :return: (negative_src_node_ids, negative_dst_node_ids)
        """
        if self.negative_sample_strategy == 'random':
            negative_src_node_ids, negative_dst_node_ids = self.random_sample(size=size)
        elif self.negative_sample_strategy == 'historical' and (batch_src_node_ids is not None and batch_dst_node_ids is not None):
            negative_src_node_ids, negative_dst_node_ids = self.historical_sample(
                size=size, batch_src_node_ids=batch_src_node_ids, batch_dst_node_ids=batch_dst_node_ids,
                current_batch_start_time=current_batch_start_time, current_batch_end_time=current_batch_end_time,
            )
        elif self.negative_sample_strategy == 'inductive' and (batch_src_node_ids is not None and batch_dst_node_ids is not None):
            negative_src_node_ids, negative_dst_node_ids = self.inductive_sample(
                size=size, batch_src_node_ids=batch_src_node_ids, batch_dst_node_ids=batch_dst_node_ids,
                current_batch_start_time=current_batch_start_time, current_batch_end_time=current_batch_end_time,
            )
        else:
            raise ValueError(f'Not implemented error for negative_sample_strategy {self.negative_sample_strategy}!')

        return negative_src_node_ids, negative_dst_node_ids

    def random_sample(self, size: int):
        """
        Sample source/destination ids independently and uniformly at random
        (the strategy used by earlier works; no collision checking).

        :param size: int
        :return: (negative_src_node_ids, negative_dst_node_ids)
        """
        if self.seed is None:
            random_sample_edge_src_node_indices = np.random.randint(0, len(self.unique_src_node_ids), size)
            random_sample_edge_dst_node_indices = np.random.randint(0, len(self.unique_dst_node_ids), size)
        else:
            random_sample_edge_src_node_indices = self.random_state.randint(0, len(self.unique_src_node_ids), size)
            random_sample_edge_dst_node_indices = self.random_state.randint(0, len(self.unique_dst_node_ids), size)

        return (self.unique_src_node_ids[random_sample_edge_src_node_indices],
                self.unique_dst_node_ids[random_sample_edge_dst_node_indices])

    def random_sample_with_collision_check(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray):
        """
        Random sampling that guarantees the sampled edges do not collide with
        edges present in the current batch. Used to top up the 'historical'
        and 'inductive' strategies when they run out of candidates.

        :param size: int
        :param batch_src_node_ids: ndarray, shape (batch_size, )
        :param batch_dst_node_ids: ndarray, shape (batch_size, )
        :return: (negative_src_node_ids, negative_dst_node_ids)
        """
        assert batch_src_node_ids is not None and batch_dst_node_ids is not None

        batch_edges = set(zip(batch_src_node_ids, batch_dst_node_ids))
        possible_random_edges = list(self.possible_edges - batch_edges)
        assert len(possible_random_edges) > 0

        # `replace=True` allows repeats when we need more edges than are available.
        random_edge_indices = self.random_state.choice(
            len(possible_random_edges), size=size, replace=len(possible_random_edges) < size
        )
        return (np.array([possible_random_edges[i][0] for i in random_edge_indices]),
                np.array([possible_random_edges[i][1] for i in random_edge_indices]))

    def historical_sample(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray,
                           current_batch_start_time: float, current_batch_end_time: float):
        """
        Prefer edges that were observed at some point in the past (before the
        current batch) but are not part of the current batch. If there
        aren't enough such edges, fill the remainder with collision-checked
        random samples.

        :param size: int
        :param batch_src_node_ids: ndarray, shape (batch_size, )
        :param batch_dst_node_ids: ndarray, shape (batch_size, )
        :param current_batch_start_time: float
        :param current_batch_end_time: float
        :return: (negative_src_node_ids, negative_dst_node_ids)
        """
        assert self.seed is not None

        historical_edges = self.get_unique_edges_between_start_end_time(
            start_time=self.earliest_time, end_time=current_batch_start_time
        )
        current_batch_edges = self.get_unique_edges_between_start_end_time(
            start_time=current_batch_start_time, end_time=current_batch_end_time
        )
        unique_historical_edges = historical_edges - current_batch_edges
        unique_historical_edges_src_node_ids = np.array([edge[0] for edge in unique_historical_edges])
        unique_historical_edges_dst_node_ids = np.array([edge[1] for edge in unique_historical_edges])

        if size > len(unique_historical_edges):
            num_random_sample_edges = size - len(unique_historical_edges)
            random_sample_src_node_ids, random_sample_dst_node_ids = self.random_sample_with_collision_check(
                size=num_random_sample_edges, batch_src_node_ids=batch_src_node_ids, batch_dst_node_ids=batch_dst_node_ids
            )

            negative_src_node_ids = np.concatenate([random_sample_src_node_ids, unique_historical_edges_src_node_ids])
            negative_dst_node_ids = np.concatenate([random_sample_dst_node_ids, unique_historical_edges_dst_node_ids])
        else:
            historical_sample_edge_node_indices = self.random_state.choice(len(unique_historical_edges), size=size, replace=False)
            negative_src_node_ids = unique_historical_edges_src_node_ids[historical_sample_edge_node_indices]
            negative_dst_node_ids = unique_historical_edges_dst_node_ids[historical_sample_edge_node_indices]

        # np.concatenate silently upcasts to float when one input is empty;
        # cast back to long so the ids remain valid array indices.
        return negative_src_node_ids.astype(np.longlong), negative_dst_node_ids.astype(np.longlong)

    def inductive_sample(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray,
                          current_batch_start_time: float, current_batch_end_time: float):
        """
        Like `historical_sample`, but additionally excludes any edge seen
        during the "observed" period (`self.observed_edges`), so evaluation
        only sees genuinely unseen negative edges.

        :param size: int
        :param batch_src_node_ids: ndarray, shape (batch_size, )
        :param batch_dst_node_ids: ndarray, shape (batch_size, )
        :param current_batch_start_time: float
        :param current_batch_end_time: float
        :return: (negative_src_node_ids, negative_dst_node_ids)
        """
        assert self.seed is not None

        historical_edges = self.get_unique_edges_between_start_end_time(
            start_time=self.earliest_time, end_time=current_batch_start_time
        )
        current_batch_edges = self.get_unique_edges_between_start_end_time(
            start_time=current_batch_start_time, end_time=current_batch_end_time
        )
        # Historical edges that are neither already-observed nor in the current batch.
        unique_inductive_edges = historical_edges - self.observed_edges - current_batch_edges
        unique_inductive_edges_src_node_ids = np.array([edge[0] for edge in unique_inductive_edges])
        unique_inductive_edges_dst_node_ids = np.array([edge[1] for edge in unique_inductive_edges])

        if size > len(unique_inductive_edges):
            num_random_sample_edges = size - len(unique_inductive_edges)
            random_sample_src_node_ids, random_sample_dst_node_ids = self.random_sample_with_collision_check(
                size=num_random_sample_edges, batch_src_node_ids=batch_src_node_ids, batch_dst_node_ids=batch_dst_node_ids
            )

            negative_src_node_ids = np.concatenate([random_sample_src_node_ids, unique_inductive_edges_src_node_ids])
            negative_dst_node_ids = np.concatenate([random_sample_dst_node_ids, unique_inductive_edges_dst_node_ids])
        else:
            inductive_sample_edge_node_indices = self.random_state.choice(len(unique_inductive_edges), size=size, replace=False)
            negative_src_node_ids = unique_inductive_edges_src_node_ids[inductive_sample_edge_node_indices]
            negative_dst_node_ids = unique_inductive_edges_dst_node_ids[inductive_sample_edge_node_indices]

        return negative_src_node_ids.astype(np.longlong), negative_dst_node_ids.astype(np.longlong)

    def reset_random_state(self):
        """Reinitialize the RNG from `self.seed` (useful before eval passes)."""
        self.random_state = np.random.RandomState(self.seed)


# --------------------------------------------------------------------------- #
# Dataset statistics
# --------------------------------------------------------------------------- #

def compute_stats(data: Data):
    """
    Compute mean/std inter-event time deltas, useful as sanity-check
    statistics or as normalization constants for time encodings.

    Tracks, for every node, the timestamp of its previous interaction (split
    by role: as a source, as a destination, and combined), and reports the
    time elapsed since then for every new interaction.

    :param data: Data
    :return: (mean_delta_t, std_delta_t, init_time)
    """
    init_time = np.min(data.node_interact_times)

    last_timestamp_src = dict()
    last_timestamp_dst = dict()
    last_timestamp = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    all_timediffs = []

    for src, dst, t in zip(data.src_node_ids, data.dst_node_ids, data.node_interact_times):
        src, dst, t = src.item(), dst.item(), t.item()

        all_timediffs_src.append(t - last_timestamp_src.get(src, init_time))
        all_timediffs_dst.append(t - last_timestamp_dst.get(dst, init_time))
        all_timediffs.append(t - last_timestamp.get(src, init_time))
        all_timediffs.append(t - last_timestamp.get(dst, init_time))

        last_timestamp_src[src] = t
        last_timestamp_dst[dst] = t
        last_timestamp[src] = t
        last_timestamp[dst] = t

    src_and_dst = all_timediffs_src + all_timediffs_dst
    mean_delta_t = np.mean(all_timediffs)
    std_delta_t = np.std(all_timediffs)

    print(f'avg delta_t(src): {np.mean(all_timediffs_src)} +/- {np.std(all_timediffs_src)}')
    print(f'avg delta_t(dst): {np.mean(all_timediffs_dst)} +/- {np.std(all_timediffs_dst)}')
    print(f'avg delta_t(src+dst): {np.mean(src_and_dst)} +/- {np.std(src_and_dst)}')
    print(f'avg delta_t(all): {mean_delta_t} +/- {std_delta_t}')

    return mean_delta_t, std_delta_t, init_time


# --------------------------------------------------------------------------- #
# Batched multi-hop temporal subgraphs
# --------------------------------------------------------------------------- #

class BatchSubgraphs:
    """
    A container for storing and manipulating a batch of temporal subgraphs
    across multiple layers (hops).

    Each layer holds, per batch instance:
        - nodes           : node ids in the subgraph
        - events          : event (edge) ids associated with each sampled neighbor
        - timestamps      : timestamps for each event
        - event_features  : feature tensors describing each event
        - event_attention  : attention weights over events
        - node_attention   : attention weights over nodes
        - timing_attention : attention weights over timestamps

    This layout is typical for models that process multi-hop temporal
    subgraphs hop-by-hop (e.g. TGAT-style architectures).

    Attributes
    ----------
    nodes : List[np.ndarray]
        Node ids per layer. Shape per layer: (batch_size, neighbors).
    events : List[np.ndarray]
        Event ids per layer. Shape per layer: (batch_size, neighbors).
    timestamps : List[np.ndarray]
        Event timestamps per layer. Shape per layer: (batch_size, neighbors).
    event_features : List[torch.Tensor]
        Event feature tensors per layer. Shape per layer: (batch_size, neighbors, feature_dim).
    event_attention : List[torch.Tensor]
        Attention masks over events, per layer.
    node_attention : List[torch.Tensor]
        Attention masks over nodes, per layer.
    timing_attention : List[torch.Tensor]
        Attention masks over timestamps, per layer.

    Raises
    ------
    AssertionError
        If layer counts or shapes between the provided parameters don't match.
    """

    def __init__(self, nodes: List[np.ndarray], events: List[np.ndarray], timestamps: List[np.ndarray],
                 event_features: List[torch.Tensor], event_attention: Optional[List[torch.Tensor]] = None,
                 node_attention: Optional[List[torch.Tensor]] = None, timing_attention: Optional[List[torch.Tensor]] = None):
        """
        Initialize a batch of multi-layer subgraphs.

        Parameters
        ----------
        nodes : List[np.ndarray]
            Node id arrays, one per layer.
        events : List[np.ndarray]
            Event id arrays, one per layer.
        timestamps : List[np.ndarray]
            Timestamp arrays, one per layer.
        event_features : List[torch.Tensor]
            Event feature tensors, one per layer.
        event_attention : Optional[List[torch.Tensor]], default=None
            Attention masks for events. Defaults to all-ones tensors.
        node_attention : Optional[List[torch.Tensor]], default=None
            Attention masks for nodes. Defaults to all-ones tensors.
        timing_attention : Optional[List[torch.Tensor]], default=None
            Attention masks for timestamps. Defaults to all-ones tensors.

        Raises
        ------
        AssertionError
            If input lists don't have the same length, or their shapes
            disagree in the first two dimensions.
        """
        assert len(nodes) == len(events) == len(timestamps) == len(event_features), (
            f"All parameters must have the same number of layers. "
            f"Found: Nodes: {len(nodes)}, events: {len(events)} "
            f"timestamps: {len(timestamps)}, event features:{len(event_features)}"
        )
        for layer in range(len(nodes)):
            assert nodes[layer].shape == events[layer].shape == timestamps[layer].shape == event_features[layer].shape[:2], (
                f"The first two dimensions of all parameters must be the same. Found in layer {layer}: "
                f"Nodes: {nodes[layer].shape}, events: {events[layer].shape} "
                f"timestamps: {timestamps[layer].shape}, event features: {event_features[layer].shape}"
            )

        self.nodes = [x.astype("int64") for x in nodes]
        self.events = [x.astype("int64") for x in events]
        self.timestamps = timestamps
        self.event_features = event_features

        if event_attention is None:
            self.event_attention = [torch.ones(x.shape) for x in events]
        else:
            assert len(event_attention) == len(timestamps), (
                f"All parameters must have the same number of layers. "
                f"Found: Timestamps: {len(timestamps)}, event attention: {len(event_attention)}"
            )
            for layer in range(len(nodes)):
                assert timestamps[layer].shape == event_attention[layer].shape, (
                    f"The first two dimensions of all parameters must be the same. Found in layer {layer}: "
                    f"Timestamps: {timestamps[layer].shape}, event attention: {event_attention[layer].shape}"
                )
            self.event_attention = event_attention

        if node_attention is None:
            self.node_attention = [torch.ones(x.shape) for x in events]
        else:
            assert len(node_attention) == len(timestamps), (
                f"All parameters must have the same number of layers. "
                f"Found: Timestamps: {len(timestamps)}, node attention: {len(node_attention)}"
            )
            for layer in range(len(nodes)):
                assert timestamps[layer].shape == node_attention[layer].shape, (
                    f"The first two dimensions of all parameters must be the same. Found in layer {layer}: "
                    f"Timestamps: {timestamps[layer].shape}, node attention: {node_attention[layer].shape}"
                )
            self.node_attention = node_attention

        if timing_attention is None:
            self.timing_attention = [torch.ones(x.shape) for x in events]
        else:
            assert len(timing_attention) == len(timestamps), (
                f"All parameters must have the same number of layers. "
                f"Found: Timestamps: {len(timestamps)}, timing attention: {len(timing_attention)}"
            )
            for layer in range(len(nodes)):
                assert timestamps[layer].shape == timing_attention[layer].shape, (
                    f"The first two dimensions of all parameters must be the same. Found in layer {layer}: "
                    f"Timestamps: {timestamps[layer].shape}, timining attention: {timing_attention[layer].shape}"
                )
            self.timing_attention = timing_attention

    # ----------------------------- slicing / equality ----------------------------- #

    def __getitem__(self, indices):
        """
        Slice the batch along the first (batch) dimension.

        Parameters
        ----------
        indices : slice
            Slice object used to extract a subset of the batch.

        Returns
        -------
        BatchSubgraphs
            A new instance containing only the sliced data.

        Raises
        ------
        AssertionError
            If `indices` is not a slice object.
        """
        assert isinstance(indices, slice), "Only slices are supported!"
        nodes = [x[indices] for x in self.nodes]
        events = [x[indices] for x in self.events]
        timestamps = [x[indices] for x in self.timestamps]
        event_features = [x[indices] for x in self.event_features]
        event_attention = [x[indices] for x in self.event_attention]
        node_attention = [x[indices] for x in self.node_attention]
        timing_attention = [x[indices] for x in self.timing_attention]

        return BatchSubgraphs(nodes, events, timestamps, event_features, event_attention, node_attention, timing_attention)

    def __eq__(self, other):
        """
        Check whether two `BatchSubgraphs` instances hold identical data.

        Parameters
        ----------
        other : BatchSubgraphs

        Returns
        -------
        bool
            True if every layer's attributes match exactly, else False.
        """
        if self.get_num_layers() != other.get_num_layers():
            return False

        if self.get_num_instances() != other.get_num_instances():
            return False

        for l in range(self.get_num_layers()):
            if (self.nodes[l] != other.nodes[l]).any():
                return False
            if (self.events[l] != other.events[l]).any():
                return False
            if (self.timestamps[l] != other.timestamps[l]).any():
                return False
            if (self.event_features[l] != other.event_features[l]).any():
                return False
            if (self.event_attention[l] != other.event_attention[l]).any():
                return False
            if (self.node_attention[l] != other.node_attention[l]).any():
                return False
            if (self.timing_attention[l] != other.timing_attention[l]).any():
                return False

        return True

    # ----------------------------- basic accessors ----------------------------- #

    def get_num_layers(self):
        """
        Returns
        -------
        int
            Number of layers (hops) in the subgraph batch.
        """
        return len(self.events)

    def get_num_instances(self):
        """
        Returns
        -------
        int
            Number of instances (batch size).
        """
        return self.events[0].shape[0]

    def get_num_events(self):
        """
        Count nonzero (i.e. real, non-padding) events per instance, summed
        across all layers.

        Returns
        -------
        np.ndarray
            Array of shape (batch_size,) with per-instance event counts.
        """
        result = np.zeros((self.get_num_instances(),))
        for e in self.events:
            result += (e != 0).sum(axis=1)
        return result

    def get_num_neighbors(self):
        """
        Returns
        -------
        int
            Neighbors-per-node width, taken from whichever of the first or
            last layer is narrower (depends on whether layers are stored in
            forward or reversed hop order).
        """
        return min(self.nodes[0].shape[1], self.nodes[-1].shape[1])

    def get_num_features(self):
        """
        Returns
        -------
        int
            Dimensionality of the event feature vectors.
        """
        return self.event_features[0].shape[2]

    def get_events(self):
        """
        Concatenate events across all layers along the neighbor dimension.

        Returns
        -------
        np.ndarray
            Shape (batch_size, total_neighbors_across_layers).
        """
        return np.concat(self.events, axis=1)

    def get_timings(self):
        """
        Concatenate timestamps across all layers along the neighbor dimension.

        Returns
        -------
        np.ndarray
            Shape (batch_size, total_neighbors_across_layers).
        """
        return np.concat(self.timestamps, axis=1)

    # ----------------------------- in-place mutation ----------------------------- #

    def set_event_attention(self, attention):
        """
        Overwrite the event attention values for every layer.

        Parameters
        ----------
        attention : List[torch.Tensor]
            Replacement attention masks, one per layer, matching the current
            `event_attention` shapes.

        Raises
        ------
        AssertionError
            If any layer's shape doesn't match the existing attention tensor.
        """
        for i, _ in enumerate(self.event_attention):
            assert self.event_attention[i].shape == attention[i].shape, (
                f"Dimensions do not match: Found {attention[i].shape} at layer {i}, "
                f"expected {self.event_attention[i].shape}"
            )
            self.event_attention[i] = attention[i]

    def chop_layers(self, new_num_layers):
        """
        Truncate the subgraph batch, keeping only the first `new_num_layers` layers.

        Parameters
        ----------
        new_num_layers : int
            Number of layers to keep.
        """
        self.nodes = self.nodes[:new_num_layers]
        self.events = self.events[:new_num_layers]
        self.timestamps = self.timestamps[:new_num_layers]
        self.event_features = self.event_features[:new_num_layers]
        self.event_attention = self.event_attention[:new_num_layers]
        self.node_attention = self.node_attention[:new_num_layers]
        self.timing_attention = self.timing_attention[:new_num_layers]

    def reverse_layers(self):
        """Reverse the order of layers in place (e.g. outermost hop first)."""
        self.nodes = self.nodes[::-1]
        self.events = self.events[::-1]
        self.timestamps = self.timestamps[::-1]
        self.event_features = self.event_features[::-1]
        self.event_attention = self.event_attention[::-1]
        self.node_attention = self.node_attention[::-1]
        self.timing_attention = self.timing_attention[::-1]

    # ----------------------------- exporting data ----------------------------- #

    def get_split(self, ignore_event_attention=False):
        """
        Bundle the core per-layer fields together.

        Parameters
        ----------
        ignore_event_attention : bool, default=False
            If True, omit `event_attention` from the returned tuple.

        Returns
        -------
        tuple
            (nodes, events, timestamps) if `ignore_event_attention` is True,
            otherwise (nodes, events, timestamps, event_attention).
        """
        if ignore_event_attention:
            return (self.nodes, self.events, self.timestamps)
        return (self.nodes, self.events, self.timestamps, self.event_attention)

    def get_split_for_layer(self, layer: int, flat_to_node=False):
        """
        Retrieve all attributes for a single layer.

        Parameters
        ----------
        layer : int
            Layer index.
        flat_to_node : bool, default=False
            If True, collapse the batch dimension into the neighbor
            dimension (useful for feeding a flat per-node batch to a model).

        Returns
        -------
        tuple
            (nodes, events, timestamps, event_features, event_attention, node_attention)
            for the requested layer.
        """
        if flat_to_node:
            return (self.nodes[layer].reshape((-1, self.get_num_neighbors())),
                    self.events[layer].reshape((-1, self.get_num_neighbors())),
                    self.timestamps[layer].reshape((-1, self.get_num_neighbors())),
                    self.event_features[layer].reshape(-1, self.get_num_neighbors(), self.event_features[layer].shape[2]),
                    self.event_attention[layer].reshape((-1, self.get_num_neighbors())),
                    self.node_attention[layer].reshape((-1, self.get_num_neighbors())))

        return (self.nodes[layer], self.events[layer], self.timestamps[layer],
                self.event_features[layer], self.event_attention[layer], self.node_attention[layer])

    def to(self, device):
        """
        Move all tensor attributes (features + attention masks) to `device`
        in place. Node/event/timestamp arrays stay on CPU as numpy arrays.

        Parameters
        ----------
        device : torch.device or str
            Target device.
        """
        self.event_features = [x.to(device) for x in self.event_features]
        self.event_attention = [x.to(device) for x in self.event_attention]
        self.node_attention = [x.to(device) for x in self.node_attention]
        self.timing_attention = [x.to(device) for x in self.timing_attention]

    # ----------------------------- masking / editing events ----------------------------- #

    def get_event_masks(self, event_id: int):
        """
        Build a per-layer boolean mask marking positions whose event id
        equals `event_id`.

        Parameters
        ----------
        event_id : int
            Event id to match.

        Returns
        -------
        List[np.ndarray]
            One boolean mask per layer, same shape as `self.events[layer]`.
        """
        result = []
        for i, e in enumerate(self.events):
            result.append(e == event_id)
        return result

    def replace_event(self, masks: list, node_attention: torch.Tensor, timing: np.ndarray, event_features: torch.Tensor):
        """
        Overwrite masked event slots with a single shared feature vector /
        node attention / timing value (i.e. all masked positions receive the
        *same* replacement).

        Parameters
        ----------
        masks : list of np.ndarray
            Per-layer boolean masks indicating which events to replace.
        node_attention : torch.Tensor
            Node attention value(s) to assign to masked events.
        timing : np.ndarray
            Timing value(s) to assign to masked events.
        event_features : torch.Tensor
            Feature vector(s) to assign to masked events.
        """
        for i, m in enumerate(masks):
            if m.any():
                self.event_features[i][m, :] = event_features
                self.node_attention[i][m] = node_attention
                self.timestamps[i][m] = timing

    def replace_event_2D(self, masks: list, node_attention: torch.Tensor, timing: np.ndarray, event_features: torch.Tensor):
        """
        Like `replace_event`, but each batch instance (row) gets its own
        replacement feature vector / node attention / timing value, rather
        than sharing one value across the whole batch.

        Parameters
        ----------
        masks : list of np.ndarray
            Per-layer boolean masks indicating which events to replace.
        node_attention : torch.Tensor
            Per-instance node attention values to assign to masked events.
        timing : np.ndarray
            Per-instance timing values to assign to masked events.
        event_features : torch.Tensor
            Per-instance feature tensors to assign to masked events.
        """
        for i, m in enumerate(masks):
            if m.any():
                for j, row in enumerate(m):
                    self.event_features[i][j, row, :] = event_features[j]
                    self.node_attention[i][j, row] = node_attention[j]
                    self.timestamps[i][j, row] = timing[j]

    def mask_event_features(self, event_id: np.ndarray, event_features: torch.Tensor):
        """
        Replace the feature vectors of every event matching `event_id`.

        Parameters
        ----------
        event_id : np.ndarray
            Event id to match.
        event_features : torch.Tensor
            Replacement feature vector.
        """
        for i, x in enumerate(self.event_features):
            mask = self.events[i] == event_id
            if mask.any():
                x[mask, :] = event_features

    def mask_event_timing(self, event_id: np.ndarray, timing):
        """
        Replace the timestamps of every event matching `event_id`, and reset
        its timing attention back to 1.0 (i.e. "trust this timestamp again").

        Parameters
        ----------
        event_id : np.ndarray
            Event id to match.
        timing : np.ndarray
            Replacement timing value.
        """
        for i, (ts, a) in enumerate(zip(self.timestamps, self.timing_attention)):
            mask = self.events[i] == event_id
            if mask.any():
                ts[mask] = timing
                a[mask] = 1.0

    def mask_node_attention(self, event_id: np.ndarray, attention: torch.Tensor):
        """
        Overwrite node attention for every event matching `event_id`.

        Parameters
        ----------
        event_id : np.ndarray
            Event id to match.
        attention : torch.Tensor
            Replacement node attention value.
        """
        for i, a in enumerate(self.node_attention):
            mask = self.events[i] == event_id
            if mask.any():
                a[mask] = attention

    def mask_events(self, event_ids: np.ndarray, event_mask: np.ndarray, data_per_event: dict):
        """
        Replace the data (timing, features) of specific events, and zero out
        their node/timing attention, using a per-event lookup table.

        Parameters
        ----------
        event_ids : np.ndarray
            Event ids to process.
        event_mask : np.ndarray
            Boolean mask, shape (batch_size, len(event_ids)), selecting which
            batch rows are relevant for which event id.
        data_per_event : dict
            Maps event id -> a tensor whose slice [1] is the timing and
            slice [2:] are the feature values to substitute in.
        """
        for i, x in enumerate(self.events):
            for j, id in enumerate(event_ids):
                col = event_mask[:, j]
                mask = np.zeros_like(x, dtype="bool")
                mask[col != 0] = x[col != 0] == id
                if mask.any():
                    self.timestamps[i][mask] = data_per_event[id][1].cpu().numpy()
                    self.event_features[i][mask, :] = data_per_event[id][2:]
                    self.node_attention[i][mask] = 0
                    self.timing_attention[i][mask] = 1.0

    def _get_default_event_array(self, layer: int, data_per_event: dict):
        """
        Build dense "default" feature/timing arrays for `layer` by looking
        up every currently-present event id in `data_per_event`. Used as the
        fallback content for events that end up getting masked out.

        Parameters
        ----------
        layer : int
            Layer index.
        data_per_event : dict
            Maps event id -> tensor where index 1 is the timing and indices
            2: onward are the feature values. Event id 0 (padding) is
            auto-populated with zeros.

        Returns
        -------
        tuple
            default_features : torch.Tensor, shape (batch_size, neighbors, feature_dim)
            default_timings : np.ndarray, shape (batch_size, neighbors)
        """
        num_features = next(iter(data_per_event.values())).shape[0]
        device = next(iter(data_per_event.values())).device
        data_per_event[0] = torch.zeros(num_features)

        default_features = torch.zeros((self.events[layer].shape[0], self.events[layer].shape[1], num_features - 2), device=device)
        default_timings = torch.zeros((self.events[layer].shape[0], self.events[layer].shape[1]), device=device)

        for k, row in enumerate(self.events[layer]):
            for j, col in enumerate(row):
                default_features[k, j, :] = data_per_event[col][2:]
                default_timings[k, j] = data_per_event[col][1]
        default_timings = default_timings.cpu().numpy()

        return default_features, default_timings

    def keep_events(self, event_ids: np.ndarray, data_per_event: Optional[dict] = None):
        """
        Restrict each row to only the events listed in `event_ids`, clearing
        (or resetting to default) everything else.

        Parameters
        ----------
        event_ids : np.ndarray
            Shape (batch_size, num_kept), event ids to keep per instance.
        data_per_event : dict, optional
            If provided, non-kept events fall back to this per-event default
            data (see `_get_default_event_array`) instead of being zeroed out.
        """
        for i, x in enumerate(self.events):
            mask = ~((x[:, :, None] == event_ids[:, None, :]).any(axis=-1))
            if data_per_event is not None:
                default_features, default_timings = self._get_default_event_array(i, data_per_event)
                if mask.any():
                    self.timestamps[i][mask] = default_timings[mask]
                    self.event_features[i][mask, :] = default_features[mask, :]
                    self.node_attention[i][mask] = 0
            elif mask.any():
                self.nodes[i][mask] = 0
                self.timestamps[i][mask] = 0
                self.event_features[i][mask, :] = 0
                self.event_attention[i][mask] = 0
                #self.node_attention[i][mask] = 0
                self.timing_attention[i][mask] = 0
                #self.events[i][mask] = 0

    def keep_features(self, kept_features: List[np.ndarray], data_per_event: Optional[dict] = None):
        """
        Fine-grained version of `keep_events`: for each instance, keep only
        specific (event_id, feature_index) cells, where feature_index == -1
        refers to the timing slot and -2 refers to the node-attention slot.
        Everything not explicitly kept is cleared (or reset to default).

        Parameters
        ----------
        kept_features : List[np.ndarray]
            Per-instance list of (event_id, feature_index) pairs to keep.
        data_per_event : dict, optional
            If provided, cleared cells fall back to this per-event default
            data instead of being zeroed out.
        """
        for i, feat in enumerate(self.event_features):
            default_features = None
            default_timings = None
            if data_per_event is not None:
                default_features, default_timings = self._get_default_event_array(i, data_per_event)

            mask_feat = torch.zeros_like(feat, dtype=torch.bool)
            mask_timing = np.zeros_like(self.events[i], dtype=bool)
            mask_node = np.zeros_like(self.events[i], dtype=bool)

            for j, l in enumerate(kept_features):
                # For every (event_id, feature_index) pair, find where that
                # event id currently sits in this row and record (position, feature_index).
                cells = []
                for cell in l:
                    c = np.where(self.events[i][j, :] == cell[0])[0].reshape((-1, 1))
                    c = np.concat([c, np.full_like(c, cell[1])], axis=1)
                    cells.extend(c)
                if len(cells) == 0:
                    continue

                cells = np.array(cells, dtype=int)
                rows = cells[:, 0]
                cols = cells[:, 1] - 2  # shift so timing == -1, node_attention == -2, features == 0, 1, 2, ...

                mask_feat[j, rows[(cols >= 0)], cols[(cols >= 0)]] = True
                mask_timing[j, rows[cols == -1]] = True
                mask_node[j, rows[cols == -2]] = True

            # Invert: True now means "not kept -> should be cleared".
            mask_feat = ~mask_feat
            mask_timing = ~mask_timing
            mask_node = ~mask_node

            if mask_feat.any():
                if default_features is not None:
                    feat[mask_feat] = default_features[mask_feat]
                else:
                    feat[mask_feat] = 0

            if mask_node.any():
                self.node_attention[i][mask_node] = 0

            if mask_timing.any():
                if default_timings is not None:
                    self.timestamps[i][mask_timing] = default_timings[mask_timing]
                else:
                    self.timestamps[i][mask_timing] = 0
                    self.timing_attention[i][mask_timing] = 0

    # ----------------------------- batch reshaping ----------------------------- #

    def repeat_nodes(self, n_times):
        """
        Repeat every instance in the batch `n_times` along the batch
        dimension (e.g. to align with `n_times` negative samples per edge).

        Parameters
        ----------
        n_times : int
            Number of repetitions.
        """
        for i, x in enumerate(self.events):
            self.nodes[i] = self.nodes[i].repeat(n_times, axis=0)
            self.events[i] = self.events[i].repeat(n_times, axis=0)
            self.timestamps[i] = self.timestamps[i].repeat(n_times, axis=0)
            self.event_features[i] = self.event_features[i].repeat(n_times, 1, 1)
            self.event_attention[i] = self.event_attention[i].repeat(n_times, 1)
            self.node_attention[i] = self.node_attention[i].repeat(n_times, 1)
            self.timing_attention[i] = self.timing_attention[i].repeat(n_times, 1)

    def split_batch(self):
        """
        Split the batch into a list of single-instance `BatchSubgraphs`.

        Returns
        -------
        List[BatchSubgraphs]
            One batch-of-size-1 `BatchSubgraphs` per original instance.
        """
        result: List[BatchSubgraphs] = []
        for i in range(self.nodes[0].shape[0]):
            sg = BatchSubgraphs(
                [x[[i]] for x in self.nodes],
                [x[[i]] for x in self.events],
                [x[[i]] for x in self.timestamps],
                [x[[i]] for x in self.event_features],
                [x[[i]] for x in self.event_attention],
                [x[[i]] for x in self.node_attention],
                [x[[i]] for x in self.timing_attention],
            )
            result.append(sg)
        return result


def concat_subgraphs(subgraphs: List[BatchSubgraphs]):
    """
    Concatenate multiple `BatchSubgraphs` (with identical layer structure)
    along the batch dimension into a single `BatchSubgraphs`.

    Parameters
    ----------
    subgraphs : List[BatchSubgraphs]
        Batches to concatenate. Must all share the same number of layers.

    Returns
    -------
    BatchSubgraphs
        A new instance containing the concatenated data.
    """
    nodes, events, timestamps = [], [], []
    event_features, event_attention, node_attention, timing_attention = [], [], [], []

    for l in range(subgraphs[0].get_num_layers()):
        nodes.append(np.concat([x.nodes[l] for x in subgraphs]))
        events.append(np.concat([x.events[l] for x in subgraphs]))
        timestamps.append(np.concat([x.timestamps[l] for x in subgraphs]))
        event_features.append(torch.concat([x.event_features[l] for x in subgraphs]))
        event_attention.append(torch.concat([x.event_attention[l] for x in subgraphs]))
        node_attention.append(torch.concat([x.node_attention[l] for x in subgraphs]))
        timing_attention.append(torch.concat([x.timing_attention[l] for x in subgraphs]))

    return BatchSubgraphs(nodes, events, timestamps, event_features, event_attention, node_attention, timing_attention)