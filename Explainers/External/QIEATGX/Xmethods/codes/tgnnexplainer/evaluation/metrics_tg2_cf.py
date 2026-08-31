from ctypes import Union
from fileinput import filename
from typing import List
import numpy as np
from pandas import DataFrame
from tqdm import tqdm
from pathlib import Path

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import math

from Xmethods.codes.tgnnexplainer.method.attn_explainer_tg import AttnExplainerTG

from Xmethods.codes.tgnnexplainer.method.subgraphx_tg import BaseExplainerTG, SubgraphXTG
from Xmethods.codes.tgnnexplainer.evaluation.metrics_tg_utils import fidelity_inv_tg, sparsity_tg


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
        
    
    def evaluate(self, explainer_results, event_idxs, ori_cfpred,factual,existed):
        event_idxs_results = []
        sparsity_results = []
        fid_inv_results = []
        fid_inv_best_results = []
        prob_results = []

        print('\nevaluating...')
        for i, (single_results, event_idx) in enumerate(zip(explainer_results, event_idxs)):
            # print(f'\nevaluate {i}th: {event_idx}')
            # self.explainer._initialize(event_idx)
            # print(self.explainer.original_scores)
            # print(self.original_scores)
            # self.explainer.tgnn_reward_wraper.compute_original_score
            # self.

            # sparsity_list, fid_inv_list, fid_inv_best_list, prob_list, best_prob, num_delete, rm_arr =  self._evaluate_one(single_results, event_idx, ori_cfpred)
            # best_prob, num_remain_nodes,remain_list,fidelity =  self._evaluate_one(single_results, event_idx, ori_cfpred,factual,existed)
            flag,fidelity,remove_use_times,len_can,best_output =  self._evaluate_one(single_results, event_idx, ori_cfpred,factual,existed)
            # import ipdb; ipdb.set_trace()
            # event_idxs_results.extend([event_idx]*len(sparsity_list))
            # sparsity_results.extend(sparsity_list)
            # fid_inv_results.extend(fid_inv_list)
            # fid_inv_best_results.extend(fid_inv_best_list)
            # prob_results.extend(prob_list)
        
        # results = {
        #     'event_idx': event_idxs_results,
        #     'sparsity': sparsity_results,
        #     'fid_inv': fid_inv_results,
        #     'fid_inv_best': fid_inv_best_results,
        #     'ori_score' : prob_results,
        #     'best_prob' : best_prob,
        #     'num_delete' : num_delete,
        #     'rm_arr' : rm_arr,
        # }
        # results = {
        #     'best_prob' : best_prob,
        #     'num_remain_nodes' : num_remain_nodes,
        #     'sparsity' : len(remain_list)/len(self.explainer.candidate_events),
        #     'fidelity' : fidelity,
        # }




        # self._save_value_results(event_idxs, results, self.suffix)
        # return results
        return flag,fidelity,remove_use_times,len_can,best_output
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
    
    def _evaluate_one(self, single_results, event_idx, ori_cfpred,factual,existed,sparsity=0.5):
        #NOTE: fidelityで変更された部分を足すor引く
        print(self.explainer.tgnn_reward_wraper.original_scores)
        tree_nodes, _ , recorder= single_results
        #　ちゃんとこいつはlogitになってた
        # tmp_pred = ori_cfpred
        # self.explainer.max_candiates

        sparsity_list = []
        pred_list = []
        len_node = []
        node_list = []

        
        candidate_events = self.explainer.candidate_events
        candidate_num = len(candidate_events)
        # print('candidate_num:',candidate_num)
        # best_prob = ori_cfpred
        if (factual & existed) or ((factual==0) & (existed==0)):
            best_prob = 0
        else: # 値を小さくしたい
            best_prob = 1
        # num_2hop = len(self.explainer.base_events)
        num_events = []
        # for i in range(1, int(sparsity*10+1)):
        #     num_events.append(max(1, int(round(candidate_num * (i / 10)))))
        # assert num_events[-1] <= candidate_num
        num_delete = 0
        fidelity = 0
        rmain_list = []
        # if factual:
        #     num_best_node = candidate_num
        # else:
        #     num_best_node = 0
        # 探索木内の全てのノードについて処理を行う
        i = -1
        # time_result = recorder['runtime']
        # result = [[] for _ in range(10)]
        # runtimes = [[] for _ in range(10)]
        max_events = self.explainer.max_candidates
        #全てのノードを評価する
        # for node in tqdm(tree_nodes, total=len(tree_nodes)):
        for node in tqdm(tree_nodes, total=len(tree_nodes)):
            # i += 1
            # for a1 in range(len(recorder['num_states'])):
            #     if i <= recorder['num_states'][a1]:
            #         _time = recorder['runtime'][a1]
            #         break
            # import ipdb; ipdb.set_trace()
            
            
            spar = sparsity_tg(node, candidate_num)
            num_use_events = len(node.coalition)
            
            if factual:
                assert num_use_events <= max_events
            else:
                # spar = 1.0 - spar      
                assert num_use_events >= max_events  

            # if factual:
            # logit_pred = self.explainer.tgnn_reward_wraper._compute_gnn_score(node.coalition, event_idx)
            # logit_pred = node.P
            ori_logit = self.explainer.tgnn_reward_wraper.original_scores
            if factual:
                if ori_logit >=0:
                    logit_pred = node.P + ori_logit
                else:
                    logit_pred = ori_logit - node.P
            else:
                if ori_logit >=0:
                    logit_pred = ori_logit - node.P
                else:
                    logit_pred = ori_logit + node.P
            
            new_pred = 1 / (1 + np.exp(- logit_pred))
            # _time = time_result[i]

            # flag = 0　#ここに書くと、最後の値しか返さない
            if factual == existed: # 値を上げる方向
                if best_prob < new_pred:
                    best_prob = new_pred                
                    fidelity = new_pred - ori_cfpred
                    if new_pred > 0.5:
                        flag = 1
                    else:
                        flag = 0
                    if factual:
                        remove_use_times = num_use_events
                    else:
                        remove_use_times = candidate_num - num_use_events
            else:
                if new_pred < best_prob:
                    fidelity = ori_cfpred - new_pred
                    if new_pred < 0.5:
                        flag = 1
                    else:
                        flag = 0
                    if factual:
                        remove_use_times = num_use_events
                    else:
                        remove_use_times = candidate_num - num_use_events
            # node.coalitionが使用しているイベントリスト
        
        return flag,fidelity,remove_use_times,candidate_num,best_prob

        for j in range(1,10):
            if (j/10) > sparsity:
                break
            if spar == (j/10):
                if result[j-1] == []:
                    result[j-1] = [flag, fidelity, num_use_events,_time]
                    runtimes[j-1] = _time
                else:
                    runtimes[j-1] = _time
                    tmp_pred = result[j-1][1]
                    if factual == existed:
                        if new_pred > tmp_pred:
                            result[j-1] = [flag, fidelity, num_use_events,_time]
                    else:
                        if new_pred < tmp_pred:
                            result[j-1] = [flag, fidelity, num_use_events,_time]                
        # print(result)
    
        if result[0] == []:
            if factual == existed:
                tmp_pred = 0
            else:
                tmp_pred = 1
            _time = runtimes[1]
            result[0] = [0,tmp_pred, candidate_num,_time]
        
        for i in range(1,10):
            if result[i] == []:
                result[i] = result[i-1]
                _time= recorder['runtime'][-1]
                result[i][3] = _time
                # runtimes[i] = runtimes[-1]
            if i == (sparsity*10):
                break
        
        # candidate_num
        # 
                        
                    

            # b_i_events = self.explainer.base_events + node.coalition
            # important_pred = self.explainer.tgnn_reward_wraper._compute_gnn_score(b_i_events, event_idx)
            # important_pred = node.P #! BUG
            # ori_pred = self.explainer.tgnn_reward_wraper.original_scores
            # fid_inv = fidelity_inv_tg(ori_pred, important_pred) # the larger the better
            

            # fid_inv = node.P
            # print(fid_inv)
            # b_i_events = self.explainer.base_events + node.coalition
            # print(b_i_events)
            # print(self.explainer.base_events)
            #
            
            # new_pred = self.explainer.tgnn_reward_wraper._compute_gnn_score(b_i_events, event_idx)
            # ここでシグモイドをかける
            # assert new_pred == node.P
            # if (new_pred > 1) or (new_pred < 0):
            #     print(new_pred)
            #     print(node.P)
            # print(new_pred)
            # exit()
            # print(node.coalition)

        #     if (factual & existed) or ((factual==0) & (existed==0)):
        #         if new_pred > best_prob:
        #             num_best_node = len(node.coalition)
        #             best_prob = new_pred
        #             rmain_list = list(node.coalition)
        #             fidelity = new_pred - ori_cfpred
        #     else: # factual & not existed or not factual & existed
        #         if new_pred < best_prob:
        #             num_best_node = len(node.coalition)
        #             best_prob = new_pred
        #             rmain_list = list(node.coalition)
        #             fidelity = ori_cfpred - new_pred
                    
        #     len_node.append(len(node.coalition))
        #     # fid_inv_list.append(fid_inv)
        #     sparsity_list.append(spar)
        #     pred_list.append(new_pred) #
        #     node_list.append(node.coalition)
        
        # num_delete = candidate_num - num_best_node
        # # return best_prob, num_best_node,rmain_list,fidelity
        return result, candidate_num


        max_length = max(len(nodes) for nodes in node_list)
        node_array = np.empty((len(node_list), max_length), dtype=object)
        for i, nodes in enumerate(node_list):
            node_array[i, :len(nodes)] = nodes
        sparsity_list = np.array(sparsity_list)
        fid_inv_list = np.array(fid_inv_list)
        pred_list = np.array(pred_list)
        len_node = np.array(len_node)
        # node_array = np.array(node_list)
        # print(fid_inv_list)

        # sort according to sparsity
        sort_idx = np.argsort(sparsity_list) # ascending of sparsity
        sort_idx2 = np.argsort(pred_list)
        sparsity_list = sparsity_list[sort_idx]
        fid_inv_list = fid_inv_list[sort_idx]
        pred_list1 = pred_list[sort_idx] #
        len_node = len_node[sort_idx] #
        sliced_nodes = node_array[sort_idx, :]
        # print('最後の値',pred_list)
        fid_inv_best = array_best(fid_inv_list)

        # ここから、確率値が高い順の代入
        sort_idx2 = np.flip(sort_idx2)
        pred_list2 = pred_list[sort_idx2] #
        # print(pred_list2[0:5])
        # exit()
        # print(pred_list2[:5])
        # print(node_array[:5])
        sliced_nodes2 = node_array[sort_idx2,:]

        # print(sliced_nodes2[:5])
        # exit()
        for i in range(len(pred_list)):
            if pred_list[-i-1] > 0:
                best_prob = pred_list[-i-1]
                num_delete = candidate_num - len_node[-i-1]
                rmain_list = list(sliced_nodes[-i-1])
        
        # print(pred_list)
        # print(sliced_nodes)

        # import ipdb; ipdb.set_trace()
        sparsity_thresholds = np.arange(0, 1.05, 0.05)
        indices = []
        for sparsity in sparsity_thresholds:
            # if sparsity == 1.0:
            #     print(np.where(sparsity_list <= sparsity)[0].max())
            indices.append( np.where(sparsity_list <= sparsity)[0].max() )


        rmain_list = []
        # ここは絶対動く
        for i in range(len(pred_list)):
            if pred_list[-i-1] > 0:
                best_prob = pred_list[-i-1]
                num_delete = candidate_num - len_node[-i-1]
                print(best_prob,num_delete)
                return sparsity_thresholds, fid_inv_list, fid_inv_best, pred_list, best_prob, num_delete,rmain_list

        num_delete = 0
        counter = 0
        pop_times = np.zeros(len(candidate_events))
        tmp_pred = 0
        # 値の高いものから順に、使われたものを保持していく
        for i in range(len(pred_list2)):
            # ここから10個選ぶ
            if i == 0:
                if pred_list2[i] > tmp_pred:
                    best_prob = pred_list2[i]
                else:
                    best_prob = tmp_pred
                    return sparsity_thresholds, fid_inv_list, fid_inv_best, pred_list, best_prob, num_delete,rmain_list
            
            if pred_list2[i] > tmp_pred: #予測値が大きくなっていれば
                counter += 1
                arr = sliced_nodes2[i]
                add_events = list(set(candidate_events)-set(arr))
                
                for a in range(len(candidate_events)):
                    for b in range(len(add_events)):
                        if add_events[b] == candidate_events[a]:
                            pop_times[a] += 1
            else:
                break
            if counter >= (candidate_num/2):
                break        
        # ここから上位10個の中の頻出の候補10個を見つける(より良いものを選ぶため)
        sorted_indices = np.argsort(pop_times)[::-1][:10]
        candidate_events = np.array(candidate_events)
        rmain_list = list(candidate_events[sorted_indices])
        return sparsity_thresholds, fid_inv_list, fid_inv_best, pred_list, best_prob, num_delete,rmain_list

        
