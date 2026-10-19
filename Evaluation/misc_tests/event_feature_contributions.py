"""Compute feature contributions for the highest-Shapley neighborhood event.

For each selected target interaction this script:

1. Computes event-level Shapley values for the target interaction.
2. Selects the neighborhood event with the largest Shapley value.
3. Computes feature-level contributions for that event.
4. Appends one CSV row per feature contribution.

Results are appended to the output file, allowing repeated runs to generate
additional rows for the selected target events.
"""

from argparse import ArgumentParser
import csv
import os

import numpy as np
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument(
    "-d", "--dataset", dest="dataset", required=True,
    help="dataset name", metavar="DATASET",
)
parser.add_argument(
    "--n_events", type=int, default=10,
    help="number of target events to process",
)
parser.add_argument(
    "--max_num_samples", type=int, default=None,
    help="maximum Monte Carlo samples for feature explanations",
)
parser.add_argument(
    "--seed", type=int, default=42,
    help="random seed used to select target events",
)
parser.add_argument(
    "--override", action="store_true", default=False,
    help="delete the existing output file before starting",
)
args = parser.parse_args()

from Config.config import CONFIG  # noqa: E402

CONFIG = CONFIG(args.dataset)

from Evaluation.model import load_model_and_data  # noqa: E402
from Explainers.Shapley4TGNN.Explainer import (  # noqa: E402
    ShapleyExplainerEvents,
    ShapleyExplainerFeatures,
)


OUTPUT_COLUMNS = [
    "target_event_id",
    "target_source",
    "target_target",
    "target_timestamp",
    "highest_event_id",
    "highest_source",
    "highest_target",
    "highest_timestamp",
    "highest_event_shapley_value",
    "feature",
    "feature_value",
    "feature_contribution",
]


def event_metadata(data, event_id: int, edge_id_to_position: dict[int, int]):
    """Return source, destination, and timestamp for an event ID."""
    # In the current datasets event IDs and array positions coincide.  The
    # mapping also keeps this script correct if that assumption changes.
    position = edge_id_to_position.get(event_id, event_id)
    return (
        int(data.src_node_ids[position]),
        int(data.dst_node_ids[position]),
        data.node_interact_times[position].item(),
    )


# Use the same model/data construction as the main evaluation pipeline.
model, full_data, _train_data, _val_data, _test_data, full_neighbor_sampler, _train_sampler, _random_sampler = load_model_and_data()
edge_raw_features = full_neighbor_sampler.edge_features.detach().cpu().numpy()

output_file = (
    f"Results/Feat_explainer_explanations/{args.dataset}/"
    "Event_vs_Feature_Explainer.csv"
)
os.makedirs(os.path.dirname(output_file), exist_ok=True)

if args.override and os.path.exists(output_file):
    os.remove(output_file)

if not os.path.exists(output_file):
    with open(output_file, "w", newline="") as stream:
        csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS).writeheader()

edge_id_to_position = {
    int(edge_id): position
    for position, edge_id in enumerate(np.asarray(full_data.edge_ids))
}

# Event IDs with labels are valid target interactions.  Preserve the existing
# convention in the repository, where event IDs index the data arrays.
rng = np.random.default_rng(args.seed)
candidate_event_ids = np.flatnonzero(~np.isnan(full_data.labels))
if args.n_events > len(candidate_event_ids):
    raise ValueError(
        f"Requested {args.n_events} events, but only "
        f"{len(candidate_event_ids)} labeled events are available."
    )
selected_event_ids = rng.choice(candidate_event_ids, size=args.n_events, replace=False)
selected_event_ids = [
    int(event_id)
    for event_id in selected_event_ids
]

print(f"Output file: {output_file}")
print(f"Processing {len(selected_event_ids)} target events.")

# First compute event-level Shapley values, then expand the top event to its
# feature-level contributions.
event_explainer = ShapleyExplainerEvents(
    model,
    full_neighbor_sampler,
    full_data,
    edge_raw_features,
    algorithm="KernelSHAP",
)
event_explainer.initialize()

feature_explainer = ShapleyExplainerFeatures(
    model,
    full_neighbor_sampler,
    full_data,
    edge_raw_features,
    feature_names=True,
    shapley_alg="MonteCarlo",
)
feature_explainer.initialize()

for target_event_id in tqdm(selected_event_ids, desc="Target events"):
    target_source, target_target, target_timestamp = event_metadata(
        full_data, target_event_id, edge_id_to_position
    )

    event_ids, shap_values = event_explainer.explain_instance(
        target_source,
        target_target,
        target_timestamp,
        silent=True,
    )
    event_ids = np.asarray(event_ids).reshape(-1)
    event_shapley_values = np.asarray(shap_values.values).reshape(-1)
    if len(event_ids) == 0:
        raise ValueError(
            f"No neighborhood events found for target event {target_event_id}."
        )

    # "Highest Shapley value" means the largest signed value, rather than the
    # event with the largest absolute attribution.
    highest_position = int(np.argmax(event_shapley_values))
    highest_event_id = int(event_ids[highest_position])
    highest_event_shapley_value = float(event_shapley_values[highest_position])
    highest_source, highest_target, highest_timestamp = event_metadata(
        full_data, highest_event_id, edge_id_to_position
    )

    explanation_flatten, _remaining_ids, _remaining_values, _baseline = (
        feature_explainer.explain_instance(
            target_source,
            target_target,
            target_timestamp,
            event_id=highest_event_id,
            max_num_samples=args.max_num_samples,
            silent=True,
        )
    )

    rows = []
    for explanation in explanation_flatten:
        # (event_id, source, target, label, timestamp, feature,
        #  feature_value, feature_contribution)
        rows.append({
            "target_event_id": target_event_id,
            "target_source": target_source,
            "target_target": target_target,
            "target_timestamp": target_timestamp,
            "highest_event_id": highest_event_id,
            "highest_source": highest_source,
            "highest_target": highest_target,
            "highest_timestamp": highest_timestamp,
            "highest_event_shapley_value": highest_event_shapley_value,
            "feature": explanation[5],
            "feature_value": float(explanation[6]),
            "feature_contribution": float(explanation[7]),
        })

    with open(output_file, "a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
        writer.writerows(rows)

print("Done generating event feature contributions.")
