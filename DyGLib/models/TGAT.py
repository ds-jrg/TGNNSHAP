import numpy as np
import torch
import torch.nn as nn

from .modules import TimeEncoder, MergeLayer, MultiHeadAttention, TGNNBackbone
from ..utils.utils import NeighborSampler, BatchSubgraphs

from torch_geometric.nn.inits import zeros, ones
from torch_geometric.utils import scatter
from typing import Callable, Optional, Dict, Tuple
from torch_geometric.nn.resolver import activation_resolver
from typing import Callable, Optional, Any, Dict, Union, List
from torch_geometric.nn import TransformerConv


class TGAT(TGNNBackbone):

    def __init__(self, num_nodes: int, node_dim: int, edge_dim: int,
                 time_feat_dim: int, dropout: float = 0.1, device: str = 'cpu', num_layers: int = 2, num_heads: int = 2,
                 edge_attention_alter_mode: str = "add"):
        """
        TGAT model.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param num_layers: int, number of temporal graph convolution layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param device: str, device
        """
        super().__init__(num_nodes, node_dim, edge_dim,
                 time_feat_dim, dropout, device)

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.is_statefull = False
        
        self.temporal_conv_layers = nn.ModuleList([MultiHeadAttention(node_feat_dim=self.node_feat_dim,
                                                                      edge_feat_dim=self.edge_feat_dim,
                                                                      time_feat_dim=self.time_feat_dim,
                                                                      num_heads=self.num_heads,
                                                                      dropout=self.dropout,
                                                                      edge_attention_alter_mode=edge_attention_alter_mode) for _ in range(num_layers)])
        # follow the TGAT paper, use merge layer to combine the attention results and node original feature
        # self.merge_layers = nn.ModuleList([MergeLayer(input_dim1=self.node_feat_dim + self.time_feat_dim, input_dim2=self.node_feat_dim,
        #                                               hidden_dim=0, output_dim=self.node_feat_dim) for _ in range(num_layers)])
        
        self.projectors = nn.ModuleList([nn.Linear(2 * self.node_feat_dim + self.time_feat_dim, self.node_feat_dim) for _ in range(num_layers)])

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray,
                                                 src_subgraphs: BatchSubgraphs, dst_subgraphs: BatchSubgraphs,                                                 
                                                 **kwargs):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        src_subgraphs.reverse_layers()
        dst_subgraphs.reverse_layers()

        src_node_embeddings = self.compute_node_temporal_embeddings(node_ids=src_node_ids, node_interact_times=node_interact_times,
                                                                    current_layer_num=self.num_layers, subgraphs=src_subgraphs, current_neighbor_layer=self.num_layers,)

        dst_node_embeddings = self.compute_node_temporal_embeddings(node_ids=dst_node_ids, node_interact_times=node_interact_times,
                                                                    current_layer_num=self.num_layers, subgraphs=dst_subgraphs, current_neighbor_layer=self.num_layers,)
        
        src_subgraphs.reverse_layers()
        dst_subgraphs.reverse_layers()

        return src_node_embeddings, dst_node_embeddings

    def compute_node_temporal_embeddings(self, node_ids: np.ndarray, node_interact_times: np.ndarray,
                                         current_layer_num: int, subgraphs: BatchSubgraphs, current_neighbor_layer: int) -> torch.Tensor:
        """
        given node ids node_ids, and the corresponding time node_interact_times,
        return the temporal embeddings after convolution at the current_layer_num
        :param node_ids: ndarray, shape (batch_size, ) or (*, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ) or (*, ), node interaction times
        :param current_layer_num: int, current layer number
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        assert current_layer_num >= 0
        device = self.node_features.device

        # query (source) node always has the start time with time interval == 0
        # Tensor, shape (batch_size, 1, time_feat_dim)
        node_time_features = self.time_encoder(timestamps=torch.zeros(node_interact_times.shape).unsqueeze(dim=1).to(device))
        # Tensor, shape (batch_size, node_feat_dim)
        node_raw_features = self.node_features[torch.from_numpy(node_ids)]

        #kept_edge_ids_neighbors = kept_edge_ids.repeat(num_neighbors, axis=0) if kept_edge_ids.size != 0 else np.array([])

        if current_layer_num == 0:
            return node_raw_features
        else:

            node_conv_features = self.compute_node_temporal_embeddings(node_ids=node_ids,
                                                                       node_interact_times=node_interact_times,
                                                                       current_layer_num=current_layer_num - 1,
                                                                       subgraphs=subgraphs, current_neighbor_layer=current_neighbor_layer)
            
            neighbor_node_ids, neighbor_edge_ids, neighbor_times, neighbor_edge_features, edge_attn, node_attn = subgraphs.get_split_for_layer(current_neighbor_layer - 1, flat_to_node=True)
            

            # get neighbor features from previous layers
            # shape (batch_size * num_neighbors, node_feat_dim)
            neighbor_node_conv_features = self.compute_node_temporal_embeddings(node_ids=neighbor_node_ids.flatten(),
                                                                                node_interact_times=neighbor_times.flatten(),
                                                                                current_layer_num=current_layer_num - 1,
                                                                                subgraphs=subgraphs,
                                                                                current_neighbor_layer=current_neighbor_layer - 1)
            # shape (batch_size, num_neighbors, node_feat_dim)
            neighbor_node_conv_features = neighbor_node_conv_features.reshape(neighbor_node_ids.shape[0], neighbor_node_ids.shape[1], self.node_feat_dim)

            neighbor_node_conv_features = self.mask_neighbors(node_attn, neighbor_node_conv_features)
            
            # compute time interval between current time and historical interaction time
            # adarray, shape (batch_size, num_neighbors)
            neighbor_delta_times = node_interact_times[:,np.newaxis] - neighbor_times

            # shape (batch_size, num_neighbors, time_feat_dim)
            neighbor_time_features = self.time_encoder(timestamps=torch.from_numpy(neighbor_delta_times).float().to(device))

            # temporal graph convolution
            # Tensor, output shape (batch_size, node_feat_dim + time_feat_dim)
            output, _ = self.temporal_conv_layers[current_layer_num - 1](node_features=node_conv_features,
                                                                         node_time_features=node_time_features,
                                                                         neighbor_node_features=neighbor_node_conv_features,
                                                                         neighbor_node_time_features=neighbor_time_features,
                                                                         neighbor_node_edge_features=neighbor_edge_features,
                                                                         neighbor_masks=neighbor_node_ids,
                                                                         edge_attn = edge_attn)

            # Tensor, output shape (batch_size, node_feat_dim)
            # follow the TGAT paper, use merge layer to combine the attention results and node original feature
            #output = self.merge_layers[current_layer_num - 1](input_1=output, input_2=node_raw_features)
            output = self.projectors[current_layer_num - 1](torch.hstack((output, node_raw_features)))

            return output

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


