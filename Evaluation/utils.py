from Config.colors import PALLETTE2, PRIMARYCOLOR
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import numpy as np
from sklearn.metrics import auc
from typing import List

def smooth_values(df, metric):
    # Step 1: create uniform time grid
    t_new = np.linspace(df["Sparsity thresholds"].min(), df["Sparsity thresholds"].max(), 200)
    
    # Step 2: interpolate values onto uniform grid
    v_interp = np.interp(t_new, df["Sparsity thresholds"], df[metric])
    
    # Step 3: smooth the uniformly spaced signal
    v_smooth = gaussian_filter1d(v_interp, sigma=2, mode='nearest')
    
    # Step 4: interpolate back to original timestamps
    result = np.interp(df["Sparsity thresholds"], t_new, v_smooth)
    return result

def load_results(dataset_name: str, as_df: bool = True) -> pd.DataFrame:
    event_df = pd.read_csv(f"Results/{dataset_name}_Shapley4TGNNEvent_agg.csv")
    feat_df = pd.read_csv(f"Results/{dataset_name}_Shapley4TGNNFeature_agg.csv")
    tgnn_df = pd.read_csv(f"Results/{dataset_name}_TGNNExplainer_agg.csv")
    tempme_df = pd.read_csv(f"Results/{dataset_name}_TempME_agg.csv")

    results_list = [event_df, feat_df, tgnn_df, tempme_df]
    if not as_df:
        return results_list
    
    df = pd.concat(results_list, ignore_index=True)
    return df

def create_diagram(df, metric: str, dataset_name: str, filename: str, silent: bool = False):
    explainer_names = {
    "Shapley4TGNNEvent": "Shapley (Event)",
    "Shapley4TGNNFeature": "Shapley (Feature)",
    "TGNNExplainer": "TGNN Explainer",
    "TempME": "TempME"
    }
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 10
    #plt.rcParams["font.serif"] = ["Times New Roman"]

    df.loc[df["Remove technique"]=="Zero", "Remove technique"] = "Removed"
    df.loc[df["Remove technique"]=="Mean", "Remove technique"] = "Replaced with avg."
    
    for group in df[["Explainer", "Remove technique"]].drop_duplicates().iterrows():
        values = df[(df["Explainer"]==group[1].iloc[0]) & (df["Remove technique"]==group[1].iloc[1])][["Sparsity thresholds", metric]]
        smothed_values = smooth_values(values, metric)
        df.loc[(df["Explainer"]==group[1].iloc[0]) & (df["Remove technique"]==group[1].iloc[1]), metric] = smothed_values

    
    #df.loc[:, metric] = gaussian_filter1d(df[metric], sigma=2, truncate=2)

    g = sns.FacetGrid(df, col="Remove technique", hue="Explainer", legend_out=True, palette=PALLETTE2, sharey=False, sharex=True)
    g.map_dataframe(sns.lineplot, x="Sparsity thresholds", y=metric)
    g.set_titles(col_template="{col_name}")

    # Add a legend below the plots, centered
    g.add_legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=2, frameon=False, title="")

    # Rename the legend labels
    for text in g._legend.texts:
        old_label = text.get_text()
        if old_label in explainer_names:
            text.set_text(explainer_names[old_label])
            
    plt.tight_layout()
    plt.gcf().set_size_inches(3.5, 1.5, forward=True)

    plt.savefig(f"Documents/Images/{dataset_name}/{filename}.pdf", bbox_inches='tight')
    if silent:
        plt.close()

def create_diagrams(df: pd.DataFrame, dataset_name: str, silent: bool = False):
    import os
    os.makedirs(f"Documents/Images/{dataset_name}", exist_ok=True)
    
    create_diagram(df, "Fidelity to prediction", dataset_name, "Fid_curve", silent=silent)
    create_diagram(df[df["Sparsity thresholds"]<=0.3], "Fidelity to prediction", dataset_name, "Fid_curve_lower_part", silent=silent)
    create_diagram(df[df["Sparsity thresholds"]>=0.7], "Fidelity to prediction", dataset_name, "Fid_curve_upper_part", silent=silent)
    
    create_diagram(df, "Deviation to ground truth", dataset_name, "Deviation_curve", silent=silent)
    create_diagram(df[df["Sparsity thresholds"]<=0.3], "Deviation to ground truth", dataset_name, "Deviation_curve_lower_part", silent=silent)
    create_diagram(df[df["Sparsity thresholds"]>=0.7], "Deviation to ground truth", dataset_name, "Deviation_curve_upper_part", silent=silent)
    
    create_diagram(df, "Fidelity to prediction (logit)", dataset_name, "Fid_logit_curve", silent=silent)
    create_diagram(df[df["Sparsity thresholds"]<=0.3], "Fidelity to prediction (logit)", dataset_name, "Fid_logit_curve_lower_part", silent=silent)
    create_diagram(df[df["Sparsity thresholds"]>=0.7], "Fidelity to prediction (logit)", dataset_name, "Fid_logit_curve_upper_part", silent=silent)
    
    create_diagram(df, "GEF", dataset_name, "GEF_curve", silent=silent)
    create_diagram(df[df["Sparsity thresholds"]<=0.3], "GEF", dataset_name, "GEF_curve_lower_part", silent=silent)
    create_diagram(df[df["Sparsity thresholds"]>=0.7], "GEF", dataset_name, "GEF_curve_upper_part", silent=silent)

def calc_auc(dataframes: List[pd.DataFrame], metric: str, remove_technique: str, no_adjust: bool = False) -> List[float]:
    """
    Compute the Area Under the Curve (AUC) for a given metric vs sparsity thresholds.

    Useful for summarizing fidelity-sparsity curves across different explanation methods.

    Returns
    -------
    list of float
        AUC values for each DataFrame in `dataframes`.
    """
    aucs = []
    minimum = min([d[metric].min() for d in dataframes])
    for d in dataframes:
        sparsites = d.loc[d["Remove technique"] == remove_technique, "Sparsity thresholds"]
        values = d.loc[d["Remove technique"] == remove_technique, metric] - minimum if not no_adjust else d.loc[d["Remove technique"] == remove_technique, metric]
        a = auc(sparsites, values)
        aucs.append(a)
    return aucs
    
if __name__ == "__main__":
    dataset_names = ["Generated" , "MOOC", "MOOCTGN", "MOOCOneHop", "Wikipedia", "Reddit"]
    for dataset_name in dataset_names:
        df = load_results(dataset_name)
        create_diagrams(df, dataset_name, silent=True)