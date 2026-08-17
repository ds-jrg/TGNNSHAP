from typing import Union
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Dataset
#from tgnnexplainer import ROOT_DIR

from ..models.ext.tgat.graph import NeighborFinder
from ..dataset.tg_dataset import verify_dataframe_unify

class MarginalSubgraphDataset(Dataset):
    """ Collect pair-wise graph data to calculate marginal contribution. """
    def __init__(self, data, exclude_mask, include_mask, subgraph_build_func) -> object:
        self.num_nodes = data.num_nodes
        self.X = data.x
        self.edge_index = data.edge_index
        self.device = self.X.device

        self.label = data.y
        self.exclude_mask = torch.tensor(exclude_mask).type(torch.float32).to(self.device)
        self.include_mask = torch.tensor(include_mask).type(torch.float32).to(self.device)
        self.subgraph_build_func = subgraph_build_func

    def __len__(self):
        return self.exclude_mask.shape[0]

    def __getitem__(self, idx):
        exclude_graph_X, exclude_graph_edge_index = self.subgraph_build_func(self.X, self.edge_index, self.exclude_mask[idx])
        include_graph_X, include_graph_edge_index = self.subgraph_build_func(self.X, self.edge_index, self.include_mask[idx])
        exclude_data = Data(x=exclude_graph_X, edge_index=exclude_graph_edge_index)
        include_data = Data(x=include_graph_X, edge_index=include_graph_edge_index)
        return exclude_data, include_data


def k_hop_temporal_subgraph(df, num_hops, event_idx):
    """
    df: temporal graph, events stream. DataFrame. An user-item bipartite graph.
    node: center user node
    num_hops: number of hops of the subgraph
    event_idx: should start from 1. 1, 2, 3, ...
    return: a sub DataFrame

    """
    # verify_dataframe(df)
    # verify_dataframe_unify(df)

    df_new = df.copy()
    df_new = df_new.rename(columns={ "idx" : "e_idx"})
    # df_new['u'] -= 1
    # df_new['i'] -= 1
    ts = df_new[df_new.e_idx == event_idx].ts.values[0]

    # center_node = df_new.iloc[event_idx-1, 0]
    src = df_new[df_new.e_idx == event_idx].u.values[0] # event_idx represents e_idx
    dst = df_new[df_new.e_idx == event_idx].i.values[0] # event_idx represents e_idx

    subsets = [np.array([src, dst]), ]
    subgraphs = [df_new[df_new.e_idx == event_idx]]
    #num_nodes = df_new[["u", "i"]].max().max() + 1

    df_new = df_new[df_new.ts < ts] # ignore events latter than event_idx

    for _ in range(num_hops):
        edges = df_new[np.isin(df_new.i, subsets[-1]) | np.isin(df_new.u, subsets[-1])]
        subsets.append(np.unique(np.concat((edges.i.values,edges.u.values))))
        subgraphs.append(edges)
        
    subgraph_df = pd.concat(subgraphs, axis=0).drop_duplicates()
    subgraph_df.index = subgraph_df.e_idx

    return subgraph_df

# def tgat_node_reindex(u: Union[int, np.array], i: Union[int, np.array], num_users: int):
#     u = u + 1
#     i = i + 1 + num_users
#     return u, i

def construct_tgat_neighbor_finder(df):
    verify_dataframe_unify(df)

    num_nodes = df['i'].max()
    adj_list = [[] for _ in range(num_nodes + 1)]
    for i in range(len(df)):
        user, item, time, e_idx = df.u[i], df.i[i], df.ts[i], df.e_idx[i]
        adj_list[user].append((item, e_idx, time))
        adj_list[item].append((user, e_idx, time))
    neighbor_finder = NeighborFinder(adj_list, uniform=False) # default 'uniform' is False

    return neighbor_finder
