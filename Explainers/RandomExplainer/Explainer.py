"""Random event-order baseline for temporal graph explanations."""

from typing import Optional

import numpy as np

from DyGLib.models.modules import TGNN
from DyGLib.utils.DataLoader import Data
from DyGLib.utils.utils import NeighborSampler
from Explainers.utils import (
    Explainer,
    compute_default_values,
    default_values_subgraph,
)

class RandomExplainer(Explainer):
    """Rank each event in a local temporal subgraph in random order.

    The baseline uses the same local event subgraphs and coalition format as
    event-level explainers, but assigns no model-based importance to events.
    """

    def __init__(
        self,
        model: TGNN,
        neighbor_finder: NeighborSampler,
        data: Data,
        event_features: np.ndarray,
        random_state: Optional[int] = None,
    ):
        super().__init__(model, neighbor_finder, data)
        self.event_features = event_features
        self._rng = np.random.default_rng(random_state)

    def initialize(self):
        """Compute the defaults required by the shared evaluation pipeline."""
        self.mean_values, self.mean_delta_timings = compute_default_values(
            self.data, self.event_features
        )

    def explain_instance(self, src, dst, timestamp, silent=False):
        """Return the local event IDs in a randomly shuffled order."""
        subgraphs_src, subgraphs_dst, event_ids, _ = default_values_subgraph(
            src,
            dst,
            timestamp,
            self.neighbor_finder,
            self.data,
            self.mean_delta_timings,
            self.mean_values,
        )
        del subgraphs_src, subgraphs_dst
        return self._rng.permutation(np.asarray(event_ids, dtype=int))

    def build_coalitions(self, explanation):
        """Build cumulative coalitions from the random event order."""
        events = np.asarray(explanation, dtype=int).reshape(-1)
        coalitions = np.zeros((len(events), len(events)))
        for i in range(len(events)):
            coalitions[i, : i + 1] = events[: i + 1]
        return coalitions, None, None


class RandomFeatureExplainer(Explainer):
    """Random baseline over event-feature players.

    Every event in the sampled computational subgraph contributes one player
    for structure, one for timing, and one for each edge feature.  The players
    are randomly ordered and returned in the same ``(event_id, feature_id)``
    format used by :class:`ShapleyExplainerFeatures`.
    """

    def __init__(
        self,
        model: TGNN,
        neighbor_finder: NeighborSampler,
        data: Data,
        event_features: np.ndarray,
        random_state: Optional[int] = None,
        top_k: Optional[int] = None,
    ):
        super().__init__(model, neighbor_finder, data)
        self.event_features = event_features
        self._rng = np.random.default_rng(random_state)
        self.is_feature_level = True
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive or None")
        self.top_k = top_k

    def initialize(self):
        """Compute the defaults required for feature masking."""
        self.mean_values, self.mean_delta_timings = compute_default_values(
            self.data, self.event_features
        )

    def explain_instance(self, src, dst, timestamp, silent=False):
        subgraphs_src, subgraphs_dst, event_ids, _ = default_values_subgraph(
            src,
            dst,
            timestamp,
            self.neighbor_finder,
            self.data,
            self.mean_delta_timings,
            self.mean_values,
        )
        del subgraphs_src, subgraphs_dst

        event_ids = self._rng.permutation(np.asarray(event_ids, dtype=int).reshape(-1))
        if self.top_k is None:
            explained_ids = event_ids
            remaining_ids = np.empty(0, dtype=int)
        else:
            explained_ids = event_ids[:self.top_k]
            remaining_ids = event_ids[self.top_k:]

        num_features = self.event_features.shape[1] + 2
        players = np.array(
            [(event_id, feature_id)
             for event_id in explained_ids
             for feature_id in range(num_features)],
            dtype=int,
        )
        return self._rng.permutation(players), remaining_ids

    def build_coalitions(self, explanation):
        """Return the random feature order as the feature importance order."""
        players, remaining_ids = explanation
        players = np.asarray(players, dtype=int)
        num_features = self.event_features.shape[1] + 2
        remaining_players = np.array(
            [(event_id, feature_id)
             for event_id in remaining_ids
             for feature_id in range(num_features)],
            dtype=int,
        )
        if len(remaining_players) == 0:
            return players, None, None
        return np.concatenate((players, remaining_players), axis=0), None, None
