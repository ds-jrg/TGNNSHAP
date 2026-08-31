import torch
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import multiprocessing as mp
from multiprocessing import Process
import numpy as np
import pandas as pd

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from dataset.tg_dataset import load_tg_dataset, load_explain_idx
from dataset.utils_dataset import construct_tgat_neighbor_finder

from TGNNmodels.xgraph.models.ext.tgat.module import TGAN
from TGNNmodels.xgraph.models.ext.tgn.model.tgn import TGN
from TGNNmodels.xgraph.models.ext.tgn.utils.data_processing import compute_time_statistics
from TGNNmodels import ROOT_DIR




def set_data(factual,lists,events,i,existed):
    tgt_src = int(lists[i][0])
    tgt_dst = int(lists[i][1])
    tgt_time = float(lists[i][2])
    tgt_index = int(lists[i][3])
    # save original event
    tgt_src1 = events.iloc[tgt_index-1,0]
    tgt_dst1 = events.iloc[tgt_index-1,1]
    tgt_time1 = events.iloc[tgt_index-1,2]
    
    if existed == 1:
        assert tgt_src1 == tgt_src
        assert tgt_dst1 == tgt_dst
        assert tgt_time1 == tgt_time
    
    return tgt_src,tgt_dst,tgt_time,tgt_index,tgt_src1,tgt_dst1,tgt_time1


def greedy_run(events0,model,lists,j,start_time,factual,existed,sparsity,config,explainer):
    
    if torch.cuda.is_available() and config.explainers.use_gpu:
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')
    one_loop=0
    len_events = len(events0)
    tgt_src,tgt_dst,tgt_time,tgt_index,tgt_src1,tgt_dst1,tgt_time1 = set_data(factual,lists,events0,j,existed)
    print(tgt_src,tgt_dst,tgt_time,tgt_index)
    events = events0.copy()
    if existed==0:
        events.iloc[tgt_index-1,0] = tgt_src
        events.iloc[tgt_index-1,1] = tgt_dst
        events.iloc[tgt_index-1,2] = tgt_time

    src_idx_l = np.array([tgt_src, ])
    target_idx_l = np.array([tgt_dst, ])
    cut_time_l = np.array([tgt_time, ])
    initial_value0 = model.get_prob( src_idx_l, target_idx_l, cut_time_l, edge_idx_preserve_list=events)
    output = initial_value0.cpu().detach().numpy()
    initial_output = output[0]


    if j == 0:
        src_idx_l1 = np.array([tgt_src1, ])
        target_idx_l1 = np.array([tgt_dst1, ])
        cut_time_l1 = np.array([tgt_time1, ])
        ori_event_score = model.get_prob( src_idx_l1, target_idx_l1, cut_time_l1,  edge_idx_preserve_list=events.index)
        ori_event_score = ori_event_score.cpu().detach().numpy()[0]
        if existed == 1:
            assert initial_output == ori_event_score #一緒かどうか確かめる
            assert initial_output > 0.5
        else:
            assert initial_output < 0.5

    if existed == 0: 
        if config.explainers.explainer_name == 'pg_explainer_tg':
            from Xmethods.codes.tgnnexplainer.method.other_baselines_tg import PGExplainerExt
            explainer = PGExplainerExt(
                                    model,
                                    config.models.model_name,
                                    config.explainers.explainer_name,
                                    config.datasets.dataset_name,
                                    events,
                                    config.explainers.param.explanation_level, 
                                    device=device,
                                    results_dir=config.explainers.results_dir,
                                    train_epochs=config.explainers.param.train_epochs,
                                    explainer_ckpt_dir=config.explainers.explainer_ckpt_dir,
                                    reg_coefs=config.explainers.param.reg_coefs,
                                    batch_size=config.explainers.param.batch_size,
                                    lr=config.explainers.param.lr,
                                    debug_mode=config.explainers.debug_mode,
            )

    explainer._initialize(tgt_index)
    candiates_2hop = explainer.candidate_events

    num_2hop = len(candiates_2hop)
    num_events = []
    for i in range(1, int(sparsity*10+1)):
        num_events.append(max(1, int(round(num_2hop * (i / 10)))))
    assert num_events[-1] <= num_2hop
    sorted_list = sorted(candiates_2hop, reverse=True) # 上位から選ぶ
    candiates = sorted_list
    sp_list = []
    
    
    if (factual == existed): 
        best_output = 0
    else: 
        best_output = 1
    
    if factual:
        used_candiates = []
    else:
        used_candiates = candiates.copy()
    loop_bests = []
    loop_list = []
    result_list = []
    one_loop_result = []
    remove_times = 0
    rm_list = []
    while 1:
        one_loop_outputs = []  
        start_time2 = time.time()      
        for i in range(len(candiates)):
            candiates_2hop = used_candiates.copy() 
            one_index = candiates[i] 
            if factual == 1:
                candiates_2hop.append(one_index) 
            else:
                candiates_2hop = [e_idx for e_idx in candiates_2hop if e_idx != one_index ]

            output2 = model.get_prob(src_idx_l, target_idx_l, cut_time_l, edge_idx_preserve_list=candiates_2hop)
            one_output = output2.cpu().detach().numpy()
            output1 = one_output[0]
            one_loop_outputs.append(output1)
            if factual:
                size_exgraph = len(candiates_2hop)
            else:
                size_exgraph = num_2hop - len(candiates_2hop)
            result_list.append([output1,size_exgraph,initial_output,num_2hop])
            one_loop_result.append([output1,one_index,initial_output,num_2hop])
            
        if one_loop == 1:
            return [],num_2hop,initial_output,result_list,[],one_loop_result
        one_loop_outputs = np.array(one_loop_outputs)  
        if (factual != existed): 
            rm_index = np.argmin(one_loop_outputs)
            loop_best = np.min(one_loop_outputs)
            loop_bests.append(loop_best)
            if loop_best >= best_output: 
                break            
            assert loop_best == one_loop_outputs[rm_index]

        else: 
            rm_index = np.argmax(one_loop_outputs)  
            loop_best = np.max(one_loop_outputs)
            loop_bests.append(loop_best)
            if loop_best <= best_output:
                break 
            assert loop_best == one_loop_outputs[rm_index]

        best_output = loop_best

        if factual: 
            rm_value = candiates[rm_index]
            used_candiates.append(rm_value)
            candiates = [e_idx for e_idx in candiates if e_idx != rm_value ]
        else: 
            rm_value = candiates[rm_index]
            rm_list.append(rm_value)
            used_candiates = [e_idx for e_idx in used_candiates if e_idx != rm_value ]
            candiates = [e_idx for e_idx in candiates if e_idx != candiates[rm_index] ]

        if factual:
            loop_len = len(used_candiates)
            candiates_str = ','.join(map(str, used_candiates))
        else:
            loop_len = len(rm_list)
            candiates_str = ','.join(map(str, rm_list))
        runtime2 = time.time() - start_time2
        loop_list.append([loop_best,loop_len,initial_output,num_2hop,runtime2,candiates_str])
        
        remove_times += 1
        if len(candiates)== 0:
            break
        if remove_times == num_events[-1]:
            _time = time.time() - start_time
            sp_list.append([0,best_output,remove_times,_time])
            break
        if remove_times in num_events: 
            _time = time.time() - start_time
            sp_list.append([0,best_output,remove_times,_time])
            
            
    if len(sp_list) == 0:
        sp_list.append([0,loop_best,remove_times,time.time()-start_time])


    if len(sp_list) < int(sparsity*10):
        loop_best = sp_list[-1][1] 
        remove_times = sp_list[-1][2]
        time_ = time.time()-start_time
        for i in range(int(sparsity*10)-len(sp_list)):
            sp_list.append([0,loop_best,remove_times,time_])

    result = sp_list
    for i in range(len(result)):
        flag = 0
        best_output = result[i][1]
        if (factual == existed):
            fidelity = best_output-initial_output
            if best_output > 0.5:
                flag = 1
        else:
            fidelity = initial_output-best_output
            if best_output < 0.5:
                flag = 1
        
        result[i][0] = flag
        result[i][1] = fidelity


    return result,num_2hop,initial_output,result_list,loop_list,one_loop_result


@hydra.main(config_path='../config', config_name='config', version_base=None)
def pipeline(config: DictConfig):
    config.models.param = config.models.param[config.datasets.dataset_name]
    config.models.ckpt_path = str(ROOT_DIR/'xgraph'/'models'/'checkpoints'/f'{config.models.model_name}_{config.datasets.dataset_name}_best.pth')
    # dataset config
    config.datasets.dataset_path = str(ROOT_DIR/'xgraph'/'dataset'/'data'/f'{config.datasets.dataset_name}.csv')
    config.datasets.explain_idx_filepath = str(ROOT_DIR/'xgraph'/'dataset'/'explain_index'/f'{config.datasets.explain_idx_filename}.csv')

    # explainer config
    config.explainers.param = config.explainers.param[config.datasets.dataset_name]
    config.explainers.results_dir = str(ROOT_DIR.parent/'benchmarks'/'results')
    config.explainers.mcts_saved_dir = str(ROOT_DIR/'xgraph'/'saved_mcts_results')
    config.explainers.explainer_ckpt_dir = str(ROOT_DIR/'xgraph'/'explainer_ckpts')


    if torch.cuda.is_available() and config.explainers.use_gpu:
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')

    # DONE: only use tgat processed data
    events, edge_feats, node_feats = load_tg_dataset(config.datasets.dataset_name)
    ngh_finder = construct_tgat_neighbor_finder(events)

    if config.models.model_name == 'tgat':
        model = TGAN(ngh_finder, node_feats, edge_feats,
                     device=device,
                     attn_mode=config.models.param.attn_mode,
                     use_time=config.models.param.use_time,
                     agg_method=config.models.param.agg_method,
                     num_layers=config.models.param.num_layers, 
                     n_head=config.models.param.num_heads,
                     num_neighbors=config.models.param.num_neighbors, 
                     drop_out=config.models.param.dropout
                     )
    elif config.models.model_name == 'tgn': # DONE: added tgn
        mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = compute_time_statistics(events.u.values, events.i.values, events.ts.values )
        model = TGN(ngh_finder, node_feats, edge_feats,
                    device=device,
                    n_layers=config.models.param.num_layers,
                    n_heads=config.models.param.num_heads,
                    dropout=config.models.param.dropout,
                    use_memory=True, # True
                    forbidden_memory_update=True, # True
                    memory_update_at_start=False, # False
                    message_dimension=config.models.param.message_dimension,
                    memory_dimension=config.models.param.memory_dimension,
                    embedding_module_type='graph_attention', # fix
                    message_function='identity', # fix
                    mean_time_shift_src=mean_time_shift_src,
                    std_time_shift_src=std_time_shift_src,
                    mean_time_shift_dst=mean_time_shift_dst,
                    std_time_shift_dst=std_time_shift_dst,
                    n_neighbors=config.models.param.num_neighbors,
                    aggregator_type='last', # fix
                    memory_updater_type='gru', # fix
                    use_destination_embedding_in_message=False,
                    use_source_embedding_in_message=False,
                    dyrep=False,
                    )
    else:    
        raise NotImplementedError('Not supported.')

    # load model checkpoints
    state_dict = torch.load(config.models.ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    if config.explainers.explainer_name == 'pg_explainer_tg':
        from Xmethods.codes.tgnnexplainer.method.other_baselines_tg_for_pg import PGExplainerExt
        explainer = PGExplainerExt(
                                model,
                                config.models.model_name,
                                config.explainers.explainer_name,
                                config.datasets.dataset_name,
                                events,
                                config.explainers.param.explanation_level, 
                                device=device,
                                results_dir=config.explainers.results_dir,
                                train_epochs=config.explainers.param.train_epochs,
                                explainer_ckpt_dir=config.explainers.explainer_ckpt_dir,
                                reg_coefs=config.explainers.param.reg_coefs,
                                batch_size=config.explainers.param.batch_size,
                                lr=config.explainers.param.lr,
                                debug_mode=config.explainers.debug_mode,
        )

    factual = int(config.factual)
    sparsity = float(config.sparse_ratio) 
    existed = int(config.existed)
    if (existed == 1) and (factual == 1):
        file_name = str(ROOT_DIR) + f"/../dataset/test_data/test_existed_fac_{config.models.model_name}_{config.datasets.dataset_name}.csv"
    elif (existed == 1) and (factual == 0):
        file_name = str(ROOT_DIR) + f"/../dataset/xgraph/test_data/test_existed_cf_{config.models.model_name}_{config.datasets.dataset_name}.csv"
    elif (existed == 0) and (factual == 1):
        file_name = str(ROOT_DIR) + f"/../dataset/xgraph/test_data/test_nonexisted_fac_{config.models.model_name}_{config.datasets.dataset_name}.csv"
    elif (existed == 0) and (factual == 0):
        file_name = str(ROOT_DIR) + f"/../dataset/xgraph/test_data/test_nonexisted_cf_{config.models.model_name}_{config.datasets.dataset_name}.csv"
    lists = np.loadtxt(file_name, delimiter=',')

    
    # save folder
    method = f'GreeDy'
    output_folder = str(ROOT_DIR) + f'/../outputs/{method}/fidelity_time/'
    loop_output_folder = str(ROOT_DIR) + f'/../outputs/{method}/loop/'
    factual_flag = 'fact' if factual else 'cf'
    existed_flag = 'existed' if existed else 'nonexisted'
    sp = int(sparsity*100)
    data = f"{config.models.model_name}_{config.datasets.dataset_name}"
    _class = f"{factual_flag}_{existed_flag}"


    random_result = [[] for _ in range(int(sparsity*10))]

    for j in range(len(lists)):
        if j == 1:
            break
        if j%10 == 1:
            print(j*10)
        start_time = time.time()
        events0 = events.copy()
        result, num_candidate,ori_cfevent_prob,result_list,loop_result,one_loop_result = greedy_run(events0,model,lists,j,start_time,factual,existed,sparsity,config,explainer)

        loop_result = np.array(loop_result,dtype=object)
        if j < 10:
            index = f"0{j}"
        else:
            index = f"{j}"
        loop_file_name = loop_output_folder + f"{_class}/{data}/sp0{sp}_{index}_{factual_flag}_{existed_flag}_{config.models.model_name}_{config.datasets.dataset_name}_loop.csv"   
        np.savetxt(loop_file_name, loop_result, fmt='%s', delimiter=',')

        for i in range(len(result)):
            random_result[i].append([result[i][0],result[i][1],result[i][2],result[i][3],num_candidate,ori_cfevent_prob])


    for i in range(int(sparsity*10)):
        j = i+1
        if factual and existed:
            file_name = output_folder + f"fact_existed/{data}/base4_{j}_factual_2hop_fidelity_{sparsity}_{config.models.model_name}_{config.datasets.dataset_name}.csv"
        elif factual and (existed==0):
            file_name = output_folder + f"fact_nonexisted/{data}/base4_{j}_factual_nonexisted_2hop_fidelity_{sparsity}_{config.models.model_name}_{config.datasets.dataset_name}.csv"
        elif (factual==0) and existed:
            file_name = output_folder + f"cf_existed/{data}/base4_cf_{j}_2hop_fidelity_{sparsity}_{config.models.model_name}_{config.datasets.dataset_name}.csv"
        elif (factual==0) and (existed==0):
            file_name = output_folder + f"cf_nonexisted/{data}/base4_cf_{j}_nonexisted_2hop_fidelity_{sparsity}_{config.models.model_name}_{config.datasets.dataset_name}.csv"
        result = np.array(random_result[i],dtype=object)
        np.savetxt(file_name, result, fmt='%f', delimiter=',')



if __name__ == '__main__':
    pipeline()


