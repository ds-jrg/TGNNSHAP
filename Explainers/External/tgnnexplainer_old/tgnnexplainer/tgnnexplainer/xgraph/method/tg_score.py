from typing import Union
from typing import List
import numpy as np
from pandas import DataFrame
import torch

from ..models.ext.tgat.module import TGAN
from ..models.ext.tgn.model.tgn import TGN
from ..evaluation.metrics_tg_utils import fidelity_inv_tg

from DyGLib.utils.utils import BatchSubgraphs, NeighborSampler
from DyGLib.models.modules import TGNN

from Config.config import CONFIG

CONFIG = CONFIG()

def _set_tgat_data(all_events: DataFrame, target_event_idx: Union[int, List]):
    """ supporter for tgat """
    if isinstance(target_event_idx, (int, np.integer)):

        edge = all_events[all_events.idx == target_event_idx]
        target_u = edge.u.values[0]
        target_i = edge.i.values[0]
        target_t = edge.ts.values[0]

        src_idx_l = np.array([target_u, ])
        target_idx_l = np.array([target_i, ])
        cut_time_l = np.array([target_t, ])
    elif isinstance(target_event_idx, (list, np.ndarray)):
        # targets = all_events[all_events.e_idx.isin(target_event_idx)]
        targets = all_events.iloc[np.isin(all_events.idx, target_event_idx)] # faster?

        target_u = targets.u.values
        target_i = targets.i.values
        target_t = targets.ts.values

        src_idx_l = target_u
        target_idx_l = target_i
        cut_time_l = target_t
    else: 
        raise ValueError

    input_data = [src_idx_l, target_idx_l, cut_time_l]
    return input_data


class TGNNRewardWraper(object):
    def __init__(self, model: Union[TGNN, TGAN, TGN], neighbor_finder: NeighborSampler, model_name, all_events, explanation_level):
        """
        """
        self.model = model
        self.neighbor_finder = neighbor_finder
        self.model_name = model_name
        self.all_events = all_events
        #self.n_users = all_events.iloc[:, 0].max() + 1
        self.explanation_level = explanation_level
        self.gamma = 0.05
        # if self.model_name == 'tgn':
            # self.tgn_memory_backup = self.model.memory.backup_memory()
    
    # def error(self, ori_pred, ptb_pred):

    #     pass

    
    def _get_model_prob(self, target_event_idx, seen_events_idxs, candidate_weights_dict=None):
        if self.model_name in ['tgat', 'tgn']:
            input_data = _set_tgat_data(self.all_events, target_event_idx)
            # seen_events_idxs = _set_tgat_events_idxs(seen_events_idxs) # NOTE: not important now
            #score = self.model.get_prob(*input_data, edge_idx_preserve_list=seen_events_idxs, logit=True)
            # score = self.model(*input_data,
            #                                     src_subgraphs = src_subgraph, dst_subgraphs = dst_subgraph, num_neighbors=20, time_gap=100)
            seen_events_idxs = np.array(seen_events_idxs).reshape(1,-1) if seen_events_idxs is not None else None

            subgraphs_src = self.neighbor_finder.get_multi_hop_neighbors(CONFIG.model.num_layers, input_data[0], input_data[2], CONFIG.model.num_neighbors, kept_edge_ids = seen_events_idxs)
            edge_feat_src = self.neighbor_finder.get_edge_features_for_multi_hop(subgraphs_src[1])
            subgraphs_src = BatchSubgraphs(*subgraphs_src, edge_feat_src)
            subgraphs_src.to(CONFIG.model.device)
            for layer_idx, (e_ids, e_attn) in enumerate(zip(subgraphs_src.events, subgraphs_src.event_attention)):
                if candidate_weights_dict is not None:
                    for e_id, new_e_attn in zip(candidate_weights_dict['candidate_events'], candidate_weights_dict['edge_weights']):
                        e_attn[e_ids==e_id.cpu()] = torch.tensor(new_e_attn, device=CONFIG.model.device)
                else:
                    subgraphs_src.event_attention[layer_idx].zero_()  # zero the stored attention mask in-place so the model actually sees it
                        
            subgraphs_dst = self.neighbor_finder.get_multi_hop_neighbors(CONFIG.model.num_layers, input_data[1], input_data[2], CONFIG.model.num_neighbors, kept_edge_ids = seen_events_idxs)
            edge_feat_dst = self.neighbor_finder.get_edge_features_for_multi_hop(subgraphs_dst[1])
            subgraphs_dst = BatchSubgraphs(*subgraphs_dst, edge_feat_dst)
            subgraphs_dst.to(CONFIG.model.device)
            for layer_idx, (e_ids, e_attn) in enumerate(zip(subgraphs_dst.events, subgraphs_dst.event_attention)):
                if candidate_weights_dict is not None:
                    for e_id, new_e_attn in zip(candidate_weights_dict['candidate_events'], candidate_weights_dict['edge_weights']):
                        e_attn[e_ids==e_id.cpu()] = torch.tensor(new_e_attn, device=CONFIG.model.device)
                else:
                    subgraphs_dst.event_attention[layer_idx].zero_()  # zero the stored attention mask in-place so the model actually sees it
            

            score = self.model(*input_data, src_subgraphs=subgraphs_src, dst_subgraphs=subgraphs_dst, time_gap=CONFIG.model.time_gap, edges_are_positive = False)
            # import ipdb; ipdb.set_trace()
        else:
            raise NotImplementedError
        
        return score.item()


    def compute_original_score(self, events_idxs, target_event_idx):
        """
        events_idxs: could be seen by model
        """
        self.original_scores = self._get_model_prob(target_event_idx, events_idxs)
        self.orininal_size = len(events_idxs)
        

    def __call__(self, events_idxs, target_event_idx):
        """
        events_idxs the all the events' indices could be seen by the gnn model. from 1
        target_event_idx is the target edge that we want to compute a reward by the temporal GNN model. from 1
        """

        if self.model_name in ['tgat', 'tgn']:
            scores = self._get_model_prob(target_event_idx, events_idxs)
            # import ipdb; ipdb.set_trace()
            reward = self._compute_reward(scores, self.orininal_size-len(events_idxs))
            return reward
        else: 
            raise NotImplementedError

    def _compute_gnn_score(self, events_idxs, target_event_idx):
        """
        events_idxs the all the events' indices could be seen by the gnn model. idxs in the all_events space, not in the tgat space.
        target_event_idx is the target edge that we want to compute a gnn score by the temporal GNN model.
        """
        return self._get_model_prob(target_event_idx, events_idxs)

        
    def _compute_reward(self, scores_petb, remove_size):
        """
        Reward should be the larger the better.
        """

        # import ipdb; ipdb.set_trace()

        if(CONFIG.model.task == "regression"):
            fid_inv = np.abs(fidelity_inv_tg(self.original_scores, scores_petb))
        else:
            fid_inv = fidelity_inv_tg(self.original_scores, scores_petb)
        return fid_inv

        # if self.original_scores >= 0:
        #     t1 = scores_petb - self.original_scores
        # else:
        #     t1 = self.original_scores - scores_petb
        
        # t2 = remove_size
        # # r = -1*t1 + -self.gamma * t2
        # # r = -t1
        # r = t1
        # return r


