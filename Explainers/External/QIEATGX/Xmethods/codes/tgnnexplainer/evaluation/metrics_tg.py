from ctypes import Union
from fileinput import filename
from typing import List
import numpy as np
from pandas import DataFrame
from tqdm import tqdm
from pathlib import Path

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))


from tgnnexplainer.xgraph.method.attn_explainer_tg import AttnExplainerTG

from tgnnexplainer.xgraph.method.subgraphx_tg import BaseExplainerTG, SubgraphXTG
from tgnnexplainer.xgraph.evaluation.metrics_tg_utils import fidelity_inv_tg, sparsity_tg


class BaseEvaluator():
    def __init__(self, model_name: str, explainer_name: str, dataset_name: str, 
                explainer: BaseExplainerTG = None,
                results_dir=None
    ) -> None:
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.explainer_name = explainer_name

        self.explainer = explainer

        self.results_dir = results_dir
        self.suffix = None


    @staticmethod
    def _save_path(results_dir, model_name, dataset_name, explainer_name, event_idxs, suffix=None):
        if isinstance(event_idxs, int):
            event_idxs = [event_idxs, ]
        
        if suffix is not None:
            filename = Path(results_dir)/f'{model_name}_{dataset_name}_{explainer_name}_{event_idxs[0]}_to_{event_idxs[-1]}_eval_{suffix}.csv'
        else:
            filename = Path(results_dir)/f'{model_name}_{dataset_name}_{explainer_name}_{event_idxs[0]}_to_{event_idxs[-1]}_eval.csv'
        return filename

    def _save_value_results(self, event_idxs, value_results, suffix=None):
        """save to a csv for plotting"""
        filename = self._save_path(self.results_dir, self.model_name, self.dataset_name, self.explainer_name, event_idxs, suffix)
        
        df = DataFrame(value_results)
        df.to_csv(filename, index=False)
        
        print(f'evaluation value results saved at {str(filename)}')

    def _evaluate_one(self, single_results, event_idx):
        raise NotImplementedError
        
    
    def evaluate(self, explainer_results, event_idxs):
        event_idxs_results = []
        sparsity_results = []
        fid_inv_results = []
        fid_inv_best_results = []
        prob_results = []

        print('\nevaluating...')
        for i, (single_results, event_idx) in enumerate(zip(explainer_results, event_idxs)):
            print(f'\nevaluate {i}th: {event_idx}')
            self.explainer._initialize(event_idx)
            # print(self.explainer.original_scores)
            # print(self.original_scores)
            # self.explainer.tgnn_reward_wraper.compute_original_score
            # self.

            sparsity_list, fid_inv_list, fid_inv_best_list, prob_list, best_prob, num_delete =  self._evaluate_one(single_results, event_idx)

            # import ipdb; ipdb.set_trace()
            event_idxs_results.extend([event_idx]*len(sparsity_list))
            sparsity_results.extend(sparsity_list)
            fid_inv_results.extend(fid_inv_list)
            fid_inv_best_results.extend(fid_inv_best_list)
            prob_results.extend(prob_list)
        
        results = {
            'event_idx': event_idxs_results,
            'sparsity': sparsity_results,
            'fid_inv': fid_inv_results,
            'fid_inv_best': fid_inv_best_results,
            'ori_score' : prob_results,
            'best_prob' : best_prob,
            'num_delete' : num_delete,
        }

        # self._save_value_results(event_idxs, results, self.suffix)
        return results
        # return ori_scores



class EvaluatorAttenTG(BaseEvaluator):
    def __init__(self, model_name: str, explainer_name: str, dataset_name: str,
                explainer: AttnExplainerTG,
                results_dir=None,
        ) -> None:
        super(EvaluatorAttenTG, self).__init__(model_name=model_name,
                                              explainer_name=explainer_name,
                                              dataset_name=dataset_name,
                                              results_dir=results_dir,
                                              explainer = explainer
                                              )
        # self.explainer = explainer

    
    # SOLVED: why 0 in the first row of results csv? sparsity calculation is wrong
    def _evaluate_one(self, single_results, event_idx):
        candidates, candidate_weights = single_results

        candidate_events = self.explainer.candidate_events
        candidate_num = len(candidate_events)
        assert len(candidates) == candidate_num

        # fid_inv_list = []
        # sparsity_list = []
        # for num in range(0, candidate_num):
        #     important_events = candidates[:num+1]
        #     b_i_events = self.explainer.base_events + important_events
        #     important_pred = self.explainer.tgnn_reward_wraper._compute_gnn_score(b_i_events, event_idx)
        #     ori_pred = self.explainer.tgnn_reward_wraper.original_scores
        #     fid_inv = fidelity_inv_tg(ori_pred, important_pred)
        #     fid_inv_list.append(fid_inv)
        #     sparsity_list.append((num+1)/candidate_num)
        #     assert np.max(sparsity_list) <= 1

        fid_inv_list = []
        sparsity_list = np.arange(0, 1.05, 0.05)
        for spar in sparsity_list:
            num = int(spar * candidate_num)
            important_events = candidates[:num+1]
            b_i_events = self.explainer.base_events + important_events
            important_pred = self.explainer.tgnn_reward_wraper._compute_gnn_score(b_i_events, event_idx)
            ori_pred = self.explainer.tgnn_reward_wraper.original_scores
            fid_inv = fidelity_inv_tg(ori_pred, important_pred)
            fid_inv_list.append(fid_inv)
            
        # import ipdb; ipdb.set_trace()
        fid_inv_best = array_best(fid_inv_list)
        sparsity = np.array(sparsity_list)
        

        return sparsity, fid_inv_list, fid_inv_best


def array_best(values):
    if len(values) == 0:
        return values
    best_values = [values[0], ]
    best = values[0]
    for i in range(1, len(values)):
        if best < values[i]:
            best = values[i]
        best_values.append(best)
    return np.array(best_values)

class EvaluatorMCTSTG(BaseEvaluator):
    def __init__(self, 
        model_name: str, explainer_name: str, dataset_name: str, 
        explainer: SubgraphXTG,
        results_dir = None
        ) -> None:
        super(EvaluatorMCTSTG, self).__init__(model_name=model_name,
                                              explainer_name=explainer_name,
                                              dataset_name=dataset_name,
                                              results_dir=results_dir,
                                            #   explainer=explainer
                                              )
        self.explainer = explainer
        self.suffix = self.explainer.suffix
        # 'pg_true' if self.explainer.pg_explainer_model is not None else 'pg_false'
    
    def _evaluate_one(self, single_results, event_idx):
        
        tree_nodes, _ = single_results
        sparsity_list = []
        fid_inv_list = []
        pred_list = []
        len_node = []
        cf_pred_list = []
        
        candidate_events = self.explainer.candidate_events
        candidate_num = len(candidate_events)
        for node in tqdm(tree_nodes, total=len(tree_nodes)):
            # import ipdb; ipdb.set_trace()
            spar = sparsity_tg(node, candidate_num)
            assert np.isclose(spar, node.Sparsity)

            # b_i_events = self.explainer.base_events + node.coalition
            # important_pred = self.explainer.tgnn_reward_wraper._compute_gnn_score(b_i_events, event_idx)
            # important_pred = node.P #! BUG
            # ori_pred = self.explainer.tgnn_reward_wraper.original_scores
            # fid_inv = fidelity_inv_tg(ori_pred, important_pred) # the larger the better
            

            fid_inv = node.P
            # print(fid_inv)
            b_i_events = self.explainer.base_events + node.coalition
            # print(b_i_events)
            # print(self.explainer.base_events)
            # exit()
            new_pred = self.explainer.tgnn_reward_wraper._compute_gnn_score(b_i_events, event_idx)
            # print(new_pred)
            # exit()

            len_node.append(len(node.coalition))
            fid_inv_list.append(fid_inv)
            sparsity_list.append(spar)
            pred_list.append(new_pred) #
        
        sparsity_list = np.array(sparsity_list)
        fid_inv_list = np.array(fid_inv_list)
        pred_list = np.array(pred_list)
        len_node = np.array(len_node)
        # print(fid_inv_list)
        # sort according to sparsity
        sort_idx = np.argsort(sparsity_list) # ascending of sparsity
        sparsity_list = sparsity_list[sort_idx]
        fid_inv_list = fid_inv_list[sort_idx]
        pred_list = pred_list[sort_idx] #
        len_node = len_node[sort_idx] #
        # print('最後の値',pred_list)
        fid_inv_best = array_best(fid_inv_list)

        # import ipdb; ipdb.set_trace()
        sparsity_thresholds = np.arange(0, 1.05, 0.05)
        indices = []
        for sparsity in sparsity_thresholds:
            # if sparsity == 1.0:
            #     print(np.where(sparsity_list <= sparsity)[0].max())
            indices.append( np.where(sparsity_list <= sparsity)[0].max() )
        
        # indices = np.unique(indices)
        # only preserve a subset of results
        # indices = np.arange(0, len(sparsity_list)+1, 5)
        # indices = np.append(indices, len(sparsity_list)-1)
        # import ipdb; ipdb.set_trace()
        # sparsity_list = sparsity_list[indices]
        # fid_inv_list = fid_inv_list[indices]
        # fid_inv_best = fid_inv_best[indices]
        # pred_list = pred_list[indices]
        # print(sparsity_thresholds)
        # print(sparsity_list)
        # print(pred_list)
        # print(len_node)
        # ori_pred = self.explainer.tgnn_reward_wraper.original_scores
        # print(ori_pred)
        # print(event_idx)
        # exit()

        # ここから、ベストな確率値を取り出す
        best_prob = -10
        num_delete = -1
        for i in range(len(pred_list)):
            if pred_list[-i-1] > 0:
                best_prob = pred_list[-i-1]
                num_delete = candidate_num - len_node[-i-1]
                print(best_prob,num_delete)
                break



        

        return sparsity_thresholds, fid_inv_list, fid_inv_best, pred_list, best_prob, num_delete