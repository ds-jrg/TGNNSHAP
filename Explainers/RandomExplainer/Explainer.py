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
