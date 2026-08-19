#!/usr/bin/env python3
"""
generate_temporal_graph.py

Generate a synthetic temporal bipartite graph containing only observed events.

Output format:
,u,i,ts,label,idx
0,1,8228,0.0,0.0,1
1,2,8229,36.0,0.0,2
...

The leading unnamed column is the CSV row index, matching the example format.

Call, e.g., 
python generate_link_pred_data.py \
    --output-dir Data/LinkPred \
  --num-users 100 \
  --num-items 500 \
  --num-events 10000 \
  --seed 42
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute a numerically stable softmax."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


def generate_dataset(
    output_dir: Path,
    num_users: int,
    num_items: int,
    num_events: int,
    item_id_offset: int,
    time_span: float,
    num_groups: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # User IDs: 1, 2, ..., num_users
    users = np.arange(1, num_users + 1, dtype=np.int64)

    # Item IDs start at item_id_offset, e.g. 8228, 8229, ...
    items = np.arange(
        item_id_offset,
        item_id_offset + num_items,
        dtype=np.int64,
    )

    # Latent user/item groups create meaningful interaction patterns.
    user_groups = rng.integers(0, num_groups, size=num_users)
    item_groups = rng.integers(0, num_groups, size=num_items)

    # Some users and items are naturally more active/popular than others.
    user_activity = rng.lognormal(mean=0.0, sigma=0.8, size=num_users)
    user_activity /= user_activity.sum()

    item_base_popularity = rng.lognormal(mean=0.0, sigma=1.0, size=num_items)
    item_base_popularity /= item_base_popularity.sum()

    # Generate increasing, irregular timestamps.
    gaps = rng.exponential(scale=time_span / num_events, size=num_events)
    timestamps = np.cumsum(gaps)
    timestamps *= time_span / timestamps[-1]

    # Temporal interaction history.
    user_recent_items: dict[int, list[int]] = defaultdict(list)
    user_item_last_seen: dict[tuple[int, int], float] = {}
    item_last_seen: dict[int, float] = {}
    dynamic_item_counts = np.ones(num_items, dtype=np.float64)

    events = []

    for idx, ts in enumerate(timestamps, start=1):
        # Optional repeating global activity cycle.
        cycle = 0.5 + 0.5 * np.sin(2 * np.pi * ts / 1000.0)

        # Choose an active user.
        user_probabilities = user_activity * (
            0.8 + 0.4 * cycle + rng.random(num_users) * 0.05
        )
        user_probabilities /= user_probabilities.sum()

        user_index = int(rng.choice(num_users, p=user_probabilities))
        user = int(users[user_index])
        user_group = user_groups[user_index]

        # Score each item for the selected user.
        scores = np.log(item_base_popularity + 1e-12)

        # Popular items become somewhat more likely to receive future events.
        scores += 0.60 * np.log1p(dynamic_item_counts)

        # Users prefer items in their latent group.
        scores[item_groups == user_group] += 2.0

        # Recently active items receive a temporal recency boost.
        for j, item in enumerate(items):
            previous_item_time = item_last_seen.get(int(item))
            if previous_item_time is not None:
                age = max(0.0, ts - previous_item_time)
                scores[j] += 0.9 * np.exp(-age / 750.0)

        # Repeated user-item interactions are likely, especially when recent.
        for j, item in enumerate(items):
            previous_pair_time = user_item_last_seen.get((user, int(item)))
            if previous_pair_time is not None:
                age = max(0.0, ts - previous_pair_time)
                scores[j] += 2.5 * np.exp(-age / 400.0)

        # With some probability, repeat one of the user's recent interactions.
        if user_recent_items[user] and rng.random() < 0.25:
            item = int(rng.choice(user_recent_items[user][-10:]))
        else:
            item_probabilities = softmax(scores)
            item = int(rng.choice(items, p=item_probabilities))

        # label=0.0 is retained to match the schema in your example.
        # Every row is an observed/positive event.
        events.append(
            {
                "u": user,
                "i": item,
                "ts": round(float(ts), 3),
                "label": 0.0,
                "idx": idx,
            }
        )

        # Update temporal state after the event.
        item_index = item - item_id_offset
        dynamic_item_counts[item_index] += 1.0
        item_last_seen[item] = float(ts)
        user_item_last_seen[(user, item)] = float(ts)

        user_recent_items[user].append(item)
        if len(user_recent_items[user]) > 30:
            user_recent_items[user] = user_recent_items[user][-30:]

    events_df = pd.DataFrame(events)

    # DyGLib expects feature row zero to be reserved for padding. Event
    # features are indexed by the one-based edge ids in the CSV.
    event_features = np.zeros((len(events_df) + 1, 2), dtype=np.float32)
    event_features[1:, 0] = events_df["ts"].to_numpy(dtype=np.float32) / max(time_span, 1.0)
    event_features[1:, 1] = events_df["idx"].to_numpy(dtype=np.float32) / max(num_events, 1)

    # Store compact node features: user/item type and normalized latent group.
    max_node_id = int(max(items.max(), users.max()))
    node_features = np.zeros((max_node_id + 1, 3), dtype=np.float32)
    node_features[users, 0] = 1.0
    node_features[users, 2] = user_groups.astype(np.float32) / max(num_groups - 1, 1)
    node_features[items, 1] = 1.0
    node_features[items, 2] = item_groups.astype(np.float32) / max(num_groups - 1, 1)

    # index=True produces the leading unnamed CSV column, as in your example.
    output_path = output_dir / "edges.csv"
    events_df.to_csv(output_path, index=True)
    np.save(output_dir / "edges.npy", event_features)
    np.save(output_dir / "edges_node.npy", node_features)

    print(f"Created {len(events_df):,} observed temporal events.")
    print(f"All events:        {output_path}")
    print(f"Event features:    {output_dir / 'edges.npy'}")
    print(f"Node features:     {output_dir / 'edges_node.npy'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a synthetic temporal graph with observed events only."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("synthetic_graph"))
    parser.add_argument("--num-users", type=int, default=500)
    parser.add_argument("--num-items", type=int, default=1000)
    parser.add_argument("--num-events", type=int, default=50_000)
    parser.add_argument("--item-id-offset", type=int, default=8228)
    parser.add_argument("--time-span", type=float, default=100_000.0)
    parser.add_argument("--num-groups", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.num_users < 1:
        raise ValueError("--num-users must be at least 1.")
    if args.num_items < 1:
        raise ValueError("--num-items must be at least 1.")
    if args.num_events < 1:
        raise ValueError("--num-events must be at least 1.")

    generate_dataset(
        output_dir=args.output_dir,
        num_users=args.num_users,
        num_items=args.num_items,
        num_events=args.num_events,
        item_id_offset=args.item_id_offset,
        time_span=args.time_span,
        num_groups=args.num_groups,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()