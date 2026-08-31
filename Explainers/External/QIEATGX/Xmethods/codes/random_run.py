import torch
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import multiprocessing as mp
from multiprocessing import Process
import numpy as np
import pandas as pd

import os, sys

from TGNNmodels import ROOT_DIR

from dataset.tg_dataset import load_tg_dataset, load_explain_idx
from dataset.utils_dataset import construct_tgat_neighbor_finder

from TGNNmodels.xgraph.models.ext.tgat.module import TGAN
from TGNNmodels.xgraph.models.ext.tgn.model.tgn import TGN
from TGNNmodels.xgraph.models.ext.tgn.utils.data_processing import compute_time_statistics


# seed
np.random.seed(0)

def set_data(factual,lists,events,i,existed):
    tgt_src = int(lists[i][0])
    tgt_dst = int(lists[i][1])
    tgt_time = float(lists[i][2])    
    tgt_index = int(lists[i][3])
    # 　save original event
    tgt_src1 = events.iloc[tgt_index-1,0]
    tgt_dst1 = events.iloc[tgt_index-1,1]
    tgt_time1 = events.iloc[tgt_index-1,2]
    
    if existed == 1:
        assert tgt_src1 == tgt_src
        assert tgt_dst1 == tgt_dst
        assert tgt_time1 == tgt_time
    
    return tgt_src,tgt_dst,tgt_time,tgt_index,tgt_src1,tgt_dst1,tgt_time1


def random_run_sp(events0,model,lists,j,factual,existed,sparsity,config,explainer,num_trials):
    if torch.cuda.is_available() and config.explainers.use_gpu:
        device = torch.device('cuda', index=config.device_id)
    else:
        AssertionError('GPU only')

    len_events = len(events0)
    tgt_src,tgt_dst,tgt_time,tgt_index,tgt_src1,tgt_dst1,tgt_time1 = set_data(factual,lists,events0,j,existed)
    
    events = events0.copy()
    if existed==0:# ターゲットイベントを代入
        events.iloc[tgt_index-1,0] = tgt_src
        events.iloc[tgt_index-1,1] = tgt_dst
        events.iloc[tgt_index-1,2] = tgt_time

    # original output
    src_idx_l = np.array([tgt_src, ])
    target_idx_l = np.array([tgt_dst, ])
    cut_time_l = np.array([tgt_time, ])
    initial_value = model.get_prob( src_idx_l, target_idx_l, cut_time_l, edge_idx_preserve_list=events.index)
    output = initial_value.cpu().detach().numpy()
    initial_output = output[0]


    if existed == 0: # for non-exist event
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

    # candidate events
    explainer._initialize(tgt_index)
    ex_candiate = explainer.candidate_events
    len_can = len(ex_candiate)

    size_min_exgraph = 1
    size_max_exgraph = max(1,int(len_can*(sparsity)))
    if size_max_exgraph == 1:
        size_min_exgraph = 0
    
    if (factual == existed):
        best_output = 0
    else:
        best_output = 1

    loop_bests = []
    size_best_exgraph = 0
    value = []     
    use_list = []
    loop_list = []
    for i in range(num_trials): # defalts500times
        start_time = time.time()
        size_exgraph = np.random.randint(size_min_exgraph,size_max_exgraph)
        if factual:    
            candiate = np.random.choice(ex_candiate, size=size_exgraph, replace=False)
        else:
            num = len_can - size_exgraph
            candiate = np.random.choice(ex_candiate, size=num, replace=False)

        start_time2 = time.time()
        output2 = model.get_prob(src_idx_l, target_idx_l, cut_time_l, edge_idx_preserve_list=candiate)
        one_output = output2.cpu().detach().numpy()
        output1 = one_output[0]
        runtime = time.time() - start_time
        if factual:
            candiates_str = ','.join(map(str, candiate))
        else:
            candiates_str = ','.join(map(str, [e_idx for e_idx in ex_candiate if e_idx not in candiate]))
        loop_list.append([output1,size_exgraph,candiates_str])
        if (factual != existed): # down prob
            if output1 < best_output:
                best_output = output1
                used_candiates = candiate.copy()
                size_best_exgraph = size_exgraph
        else: # up prob
            if output1 > best_output:
                best_output = output1
                used_candiates = candiate.copy()
                size_best_exgraph = size_exgraph

    flag = 0 # for validity
    # calculate best output
    if (factual & existed) or ((factual==0) & (existed==0)):
        fidelity = best_output-initial_output
        if best_output > 0.5:
            flag = 1
    else:
        fidelity = initial_output-best_output
        if best_output < 0.5:
            flag = 1
             
    
    return flag,fidelity,size_best_exgraph,len_can,best_output,initial_output,loop_list



@hydra.main(config_path='../config', config_name='config', version_base=None)
def pipeline(config: DictConfig):

    config.models.param = config.models.param[config.datasets.dataset_name]

    config.models.ckpt_path = str(ROOT_DIR) + '/xgraph/models/checkpoints/'+f'{config.models.model_name}_{config.datasets.dataset_name}_best.pth'
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
        assert False, 'GPU only'
        # device = torch.device('cpu')

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
    
    ###### random  #######
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

    random_result = []
    num_trials = int(config.n_trials)
    
    # save folder
    method = f'random'
    output_folder = str(ROOT_DIR) + f'/../outputs/{method}/fidelity_time/'
    loop_output_folder = str(ROOT_DIR) + f'/../outputs/{method}/loop/'
    factual_flag = 'fact' if factual else 'cf'
    existed_flag = 'existed' if existed else 'nonexisted'
    sp = int(sparsity*100)
    data = f"{config.models.model_name}_{config.datasets.dataset_name}"
    _class = f"{factual_flag}_{existed_flag}"

    for j in range(len(lists)):
        if j == 1:
            break
        start_time = time.time()
        events0 = events.copy()
        flag,fidelity,num_xgraph,num_candidate,cf_score,ori_cfevent_prob,loop_list = random_run_sp(events0,model,lists,j,factual,existed,sparsity,config,explainer,num_trials)

        runtime = time.time() - start_time
        random_result.append([flag,fidelity,num_xgraph,runtime,num_candidate,cf_score,ori_cfevent_prob])

        loop_result = np.array(loop_list,dtype=object)
        if j < 10:
            index = f"0{j}"
        else:
            index = f"{j}"
        loop_file_name = loop_output_folder + f"{_class}/{data}/sp0{sp}_n{num_trials}_{index}_{factual_flag}_{existed_flag}_{config.models.model_name}_{config.datasets.dataset_name}_loop.csv"
        np.savetxt(loop_file_name, loop_result, fmt='%s', delimiter=',')

    j = f"{int(sparsity * 100):03d}"
    if factual and existed:
        file_name = output_folder + f"fact_existed/{data}/sp{j}_factual_{method}_fidelity_{sparsity}_{config.models.model_name}_{config.datasets.dataset_name}.csv"
    elif factual & (existed==0):
        file_name = output_folder + f"fact_nonexisted/{data}/sp{j}_factual_nonexisted_{method}_fidelity_{sparsity}_{config.models.model_name}_{config.datasets.dataset_name}.csv"
    elif (factual==0) & existed:
        file_name = output_folder + f"cf_existed/{data}/sp{j}_cf_{method}_fidelity_{sparsity}_{config.models.model_name}_{config.datasets.dataset_name}.csv"
    elif (factual==0) & (existed==0):
        file_name = output_folder + f"cf_nonexisted/{data}/sp{j}_cf_nonexisted_{method}_fidelity_{sparsity}_{config.models.model_name}_{config.datasets.dataset_name}.csv"
    
    result = np.array(random_result,dtype=object)

    np.savetxt(file_name, result, fmt='%f', delimiter=',')
    print('save:',file_name)



if __name__ == '__main__':
    pipeline()


