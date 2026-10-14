"""
This module provides functions to prepare Shapley feature attribution data
and visualize it as a horizontal waterfall plot grouped by edges/events.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Optional
from datetime import timedelta, date

# Color configuration
CMAP = mpl.colormaps['RdBu']
COLORS = CMAP(np.linspace(0, 1, 5))
BLUE = COLORS[4]
LIGHTBLUE = COLORS[3]
RED = COLORS[0]
LIGHTRED = COLORS[1]
GREY = "grey"
FONTCOLOR_SUM = "white"
FONTCOLOR = "black"


def _prepare_dataset(explanation_flatten: list, k: int, base_date:Optional[date] = None) -> pd.DataFrame:
    """
    Process flattened feature-level explanations into a structured DataFrame
    for plotting.

    This performs the following steps:
    - Creates a DataFrame from the flattened explanation list.
    - Filters out zero-valued Shapley attributions.
    - Computes absolute Shapley values for ranking.
    - For each edge, retrieves the top-k features by absolute Shapley value.
    - Aggregates remaining features into 'Others'.
    - Adds a 'Sum' row representing the total Shapley value for the edge.
    - Adds convenience columns for plotting, including labels and flags.

    Parameters
    ----------
    explanation_flatten : list
        List of tuples `(Edge, Edge_type, Timing, Feat_name, Feat_value, Shap_value)`.
        Usually produced by the feature-level Shapley explainer.
    k : int
        Number of top-ranked features to include individually for each edge.
        Remaining features are aggregated into 'Others'.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with one row per feature/aggregate entry for each edge.
        Columns include:
            - 'Edge', 'Edge_type', 'Timing', 'Feat_name', 'Feat_value', 'Shap_value'
            - 'Shap_value_abs' : absolute Shapley value
            - 'IsOthers' : True if row is aggregated 'Others'
            - 'IsSum' : True if row is total sum
            - 'Label_feat' : label for y-axis in plots
    """
    # Construct initial DataFrame
    df = pd.DataFrame(
        explanation_flatten,
        columns=["Edge", "Src", "Dst", "Edge_type", "Timing", "Feat_name", "Feat_value", "Shap_value"]
    )

    # Remove zero contributions
    df = df[df.Shap_value != 0]
    df["Shap_value"] = df["Shap_value"].astype(float)

    # Convert edge to category and store absolute Shapley value
    df.Edge = df.Edge.astype("category")
    df["Shap_value_abs"] = df["Shap_value"].astype(float).abs()

    # Sum Shapley values by edge to compute total contribution
    df_sum = df.groupby("Edge", observed=False).sum()
    df_sum[["Timing", "Feat_name", "Feat_value"]] = [0, "Sum", 0]
    df_sum = df_sum.reset_index(names=["Edge"])
    df_sum = df_sum.drop(columns=["Edge_type", "Timing", "Src", "Dst"])
    df_sum = df_sum.merge(df[["Edge", "Edge_type", "Timing", "Src", "Dst"]].drop_duplicates(),
                          on="Edge", how="left")

    # Extract top-k features by absolute Shapley value for each edge
    df_top_k_per_edge = (
        df.sort_values(['Shap_value_abs'], ascending=False)
          .groupby('Edge', observed=False)
          .head(k)
    )

    # Identify "other" features not in top-k and sum their Shapley values
    others = df.merge(df_top_k_per_edge.drop_duplicates(),
                      on=['Edge', 'Feat_name'], how='left', indicator=True)
    others = df[others["_merge"] == "left_only"]
    others = others.groupby("Edge", observed=False).sum()
    others[["Timing", "Feat_name", "Feat_value"]] = [0, "Others", 0]
    others = others.reset_index(names="Edge")
    others = others.drop(columns=["Edge_type", "Timing"])
    others = others.merge(df[["Edge", "Edge_type", "Timing"]].drop_duplicates(),
                          on="Edge", how="left")

    # Combine top-k, Others, and Sum into final dataset
    df_top_k_per_edge = pd.concat([df_top_k_per_edge, others, df_sum]).reset_index(drop=True)

    # Flags for plot styling
    df_top_k_per_edge["IsOthers"] = df_top_k_per_edge.Feat_name == "Others"
    df_top_k_per_edge["IsSum"] = df_top_k_per_edge.Feat_name == "Sum"

    # Sorting for waterfall visualization
    df_top_k_per_edge = df_top_k_per_edge.sort_values(
        ["Timing", "Edge", "IsOthers", "IsSum", "Shap_value_abs"],
        ascending=[True, True, True, False, False]
    ).reset_index(drop=True)

    # Label column for y-axis
    df_top_k_per_edge["Label_feat"] = df_top_k_per_edge.apply(
        lambda x: f"{x.Feat_value} = {x.Feat_name}", axis=1
    )

    # Special labels for 'Sum' and 'Others'
    df_top_k_per_edge.loc[df_top_k_per_edge.IsSum, "Label_feat"] = \
        df_top_k_per_edge[df_top_k_per_edge.IsSum].apply(
            lambda x: f"{x.Src} to {x.Dst} @ {float(x.Timing) if base_date is None else base_date + timedelta(days=x.Timing)}", axis=1
        )
    df_top_k_per_edge.loc[df_top_k_per_edge.IsOthers, "Label_feat"] = "Others"

    return df_top_k_per_edge


def waterfall(
    explanation_flatten: list,
    baseline: float,
    k: int,
    src: Optional[str] = None,
    dst: Optional[str] = None,
    ts: Optional[str] = None,
    target: Optional[str] = None,
    remaining_ids: Optional[int] = None,
    remaining_shapley_values: Optional[float] = None,
    bar_width = 0.5,
    head_length = 0.01,
    base_date:Optional[date] = None,
    space_per_bar = 0.5
):
    """
    Create a horizontal waterfall plot from feature-level Shapley explanations.

    The plot shows for each edge:
    - The baseline prediction value.
    - Increments/decrements from individual top-k features.
    - An 'Others' bar aggregating remaining features.
    - A 'Sum' bar showing the total effect of the edge.

    Parameters
    ----------
    explanation_flatten : list
        Flattened explanation data from the feature-level Shapley explainer.
        Format: `(Edge, Edge_type, Timing, Feat_name, Feat_value, Shap_value)`.
    baseline : float
        Base prediction value before adding feature contributions.
    k : int
        Number of top features to display per edge.
    src : Optional[str], default=None
        Source node identifier for plot title.
    dst : Optional[str], default=None
        Destination node identifier for plot title.
    ts : Optional[str], default=None
        Timestamp string for plot title.
    target : Optional[str], default=None
        Target class/label for plot title.
    remaining_ids : Optional[int], default=None
        Optional remaining event IDs that have not been split up
    remaining_shapley_values : Optional[float], default=None
        Optional Shapley values of the remaining event IDs that have not been split up
    bar_width : float, default = 0.5 
        Height of the bars 
    head_length : float, default = 0.1 
        Length of the bar heads

    Returns
    -------
    matplotlib.figure.Figure
        The Matplotlib figure object containing the plot.

    Notes
    -----
    Color scheme:
        - Positive 'Sum' values: RED
        - Negative 'Sum' values: BLUE
        - Positive feature values: LIGHTRED (semi-transparent)
        - Negative feature values: LIGHTBLUE (semi-transparent)
    """
    # Prepare dataset with top-k, Others, and Sum
    df_top_k_per_edge = _prepare_dataset(explanation_flatten, k, base_date)
    if(remaining_shapley_values is not None and remaining_ids is not None):
        # Add to last row as "Other events"
        df_top_k_per_edge.loc[-1] = [0,0,0, "", 0, "Other events", 0.0, np.sum(remaining_shapley_values), np.abs(np.sum(remaining_shapley_values)), True, True, "Other events"]  # adding a row
        # Update index
        df_top_k_per_edge = df_top_k_per_edge.reset_index(drop=True) 
    num_edges = df_top_k_per_edge.Edge.drop_duplicates().shape[0]
    num_values = df_top_k_per_edge.shape[0]

    fig, ax = plt.subplots(figsize=(8, space_per_bar * k * num_edges))

    start = baseline    # running prediction for individual feature rows
    start_sum = baseline  # running prediction for 'Sum' rows

    # Calculate scaling for arrowhead width
    hl_scaled = head_length * df_top_k_per_edge.Shap_value.abs().max()
    
    y_positions = []
    y = num_values

    # Plot bars and arrows
    for i, row in df_top_k_per_edge.sort_index(ascending=False).reset_index(drop=True).sort_index(ascending=False).iterrows():
        i = float(i)  # type: ignore index cast

        if row.IsSum:
            # Sum row: total effect of edge
            y = y - 1.1
            y_positions.append(y)
            color = RED if row.Shap_value > 0 else BLUE
            ax.arrow(
                start_sum, y, row.Shap_value, 0,
                length_includes_head=True,
                color=color, width=bar_width,
                head_width=bar_width,
                head_length=min(hl_scaled, np.abs(row.Shap_value))
            )
            ax.text(
                start_sum + 0.5 * row.Shap_value,
                y,
                "{:.2f}".format(row.Shap_value),
                ha="center", va="center",
                color=FONTCOLOR_SUM
            )
            start_sum += row.Shap_value
            if row.IsOthers:
                start += row.Shap_value
        else:
            # Individual feature row
            y = y - 0.6
            y_positions.append(y)
            color = LIGHTRED if row.Shap_value > 0 else LIGHTBLUE
            ax.arrow(
                start, y, row.Shap_value, 0,
                length_includes_head=True,
                alpha=0.5, color=color, width=bar_width * 0.7,
                head_width=bar_width * 0.7,
                head_length=min(hl_scaled, np.abs(row.Shap_value))
            )
            ax.text(
                start + 0.5 * row.Shap_value,
                y,
                "{:.2f}".format(row.Shap_value),
                ha="center", va="center",
                color=FONTCOLOR
            )
            start += row.Shap_value

    # Add vertical line at baseline
    ax.vlines(x=baseline, color=GREY, linestyle="--", linewidth=1, ymin=y_positions[0] - 0.5, ymax=y_positions[0] + 0.5)
    # Add text for baseline
    ax.text(
        baseline,
        y_positions[0] + 0.5,
        "E[X]={:.2f}".format(baseline),
        verticalalignment="bottom", horizontalalignment="center",
        color=GREY
    )
    
    # Add vertical line at final prediction
    ax.vlines(x=start_sum, color=GREY, linestyle="--", linewidth=1, ymin=y_positions[-1] - 0.5, ymax=y_positions[-1] + 0.5)
    # Add text for final prediction
    ax.text(
        start_sum,
        y_positions[-1] - 0.5,
        "f(x)={:.2f}".format(start_sum),
        verticalalignment="top", horizontalalignment="center",
        color=GREY
    )
    
    #Remove borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    #Add vertical grid lines
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.1)
    ax.set_axisbelow(True)

    # Title and y-axis labels
    ax.set_title(f"Prediction from Node {src} to Node {dst} @ {ts if base_date is None else base_date + timedelta(days=ts)} (Target: {target})")
    ax.set_yticks(sorted(y_positions))
    ax.set_yticklabels(df_top_k_per_edge.Label_feat.iloc[::-1])
    ax.set_ylim(y_positions[-1] - 1.5, y_positions[0] + 1.5)

    # Bold font for sum rows
    for label, isSum in zip(ax.get_yticklabels(), df_top_k_per_edge.IsSum.iloc[::-1]):
        label.set_fontweight('bold' if isSum else 'normal')

    return fig