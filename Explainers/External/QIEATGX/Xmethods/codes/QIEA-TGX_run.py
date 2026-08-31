import torch
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import multiprocessing as mp
from multiprocessing import Process

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from dataset.tg_dataset import load_tg_dataset, load_explain_idx

from dataset.tg_dataset import load_tg_dataset, load_explain_idx
from dataset.utils_dataset import construct_tgat_neighbor_finder

from TGNNmodels.xgraph.models.ext.tgat.module import TGAN
from TGNNmodels.xgraph.models.ext.tgn.model.tgn import TGN
from TGNNmodels.xgraph.models.ext.tgn.utils.data_processing import compute_time_statistics
from TGNNmodels import ROOT_DIR
import numpy as np
import random
from functions.QIEA_func import *
import csv

def get_setting(factual,existed,model_name,dataset_name):
    if factual:
        factual_flag = 'fac'
    else:
        factual_flag = 'cf'
    if existed:
        existed_flag = 'existed'
    else:
        existed_flag = 'nonexisted'
    _class = f"{factual_flag}_{existed_flag}"
    if dataset_name == 'wikipedia':
        data = f"{model_name}_wiki"
    else:
        data = f"{model_name}_redi"
    return _class, data


def set_data(factual,lists,events,i,existed):
    tgt_src = int(lists[i][0])
    tgt_dst = int(lists[i][1])
    tgt_time = float(lists[i][2])
    tgt_index = int(lists[i][3])

    # save original event
    tgt_src1 = events.iloc[tgt_index-1,0]
    tgt_dst1 = events.iloc[tgt_index-1,1]
    tgt_time1 = events.iloc[tgt_index-1,2]
    tgt_src2 = events.iloc[tgt_index,0]
    if existed == 1:
        assert tgt_src1 == tgt_src
        assert tgt_dst1 == tgt_dst
        assert tgt_time1 == tgt_time
    
    return tgt_src,tgt_dst,tgt_time,tgt_index,tgt_src1,tgt_dst1,tgt_time1

def get_hops(_class,data,index,config):
    folder_path = str(ROOT_DIR) + f"/../dataset/hops/"
    folder_path = folder_path + f"{_class}/{data}/"
    for filename in os.listdir(folder_path):
        if index in filename:
            file = filename
            break
    hops = []
    # for file in files:
    with open(os.path.join(folder_path, file)) as f:
        reader = csv.reader(f)
        # print(reader)
        for row in reader:
            hop = []
            for i in range(len(row)):
                hop.append(int(float(row[i])))
            assert len(hop) <= 11
            hops.append(hop) 
    
    hops = np.array(hops) 
    return hops


def EA(explainer,model,up_down,factual,lists,events,j,existed,config,pg_index,seed,agents_iteration,_flag):
    if torch.cuda.is_available() and config.explainers.use_gpu:
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')


    tgt_src,tgt_dst,tgt_time,tgt_index,tgt_src1,tgt_dst1,tgt_time1 = set_data(factual,lists,events,j,existed)
    if existed==0:# ターゲットイベントを代入
        events.iloc[tgt_index-1,0] = tgt_src
        events.iloc[tgt_index-1,1] = tgt_dst
        events.iloc[tgt_index-1,2] = tgt_time

    src_idx_l = np.array([tgt_src, ])
    target_idx_l = np.array([tgt_dst, ])
    cut_time_l = np.array([tgt_time, ])
    input_data = (src_idx_l, target_idx_l, cut_time_l)
    initial_value0 = model.get_prob( src_idx_l, target_idx_l, cut_time_l, edge_idx_preserve_list=events)
    output = initial_value0.cpu().detach().numpy()
    initial_output = output[0]

    if existed == 1:
        assert initial_output  > 0.5
    else:
        assert initial_output < 0.5   

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

    explainer._initialize(tgt_index)
    candidates = explainer.candidate_events
    num_2hop = len(candidates)        

    GENOMS = num_2hop
    GENERATIONS = int(config.n_trials) # ここが試行回数
    H_gate = 0.01
    CROSSOVER_PB = 0.5
    MUTATION_PB = 0.1
    sparsity = float(config.sparse_ratio)
    teta = 0.01*np.pi

    max_events = max(1,int(sparsity*num_2hop))
    if _flag == 0:
        flag = 'ob_hops'
    elif _flag == 1:
        flag = 'ob_half'
    elif _flag == 2:
        flag = 'ob_hop_timeXtime'
    elif _flag == 3:
        flag = 'ob_hop_timeXtime_half'
    elif _flag == 4:
        flag = 'ob_hopXtime'
    elif _flag == 5:
        flag = 'ob_hop_all'
    else:
        flag = 'none'

    if j < 10:
        index = f"0{j}"
    else:
        index = f"{j}"
    _class, data = get_setting(factual,existed,config.models.model_name,config.datasets.dataset_name)
    hops = get_hops(_class,data,index,config)
    result,result_list,loop_list,model_time = Quantum_individual_optimizationA(agents_iteration,model, input_data, candidates, H_gate, teta,factual, existed,max_events,initial_output,flag,hops)


    flag = 0
    best_output = result[1]
    if (factual == existed):
        fidelity = best_output-initial_output
        if best_output > 0.5:
            flag = 1
    else:
        fidelity = initial_output-best_output
        if best_output < 0.5:
            flag = 1
    
    result[0] = flag
    result[1] = fidelity

    return result,num_2hop,initial_output,loop_list,model_time



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
    
    # load targets
    factual = int(config.factual)
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

    original_events = events.copy()
    values = []
    results = []
    sparsity = float(config.sparse_ratio)
    up_down = int(factual == existed)
    n_trials = int(config.n_trials)
    seed = int(config.seed)
    agents = int(config.QIEA_agents)
    agents_iteration = [agents,int(n_trials/agents)]


    np.random.seed(seed)
    pg_index = 0
    flag = 0
    if int(config.func_index) != 0:
        # QIEA-TGX : flag=1
        flag = int(config.func_index) 
    else:
        AssertionError()
    
    # save folder
    method = f"QIEA-TGX"
    output_folder = str(ROOT_DIR) + f'/../outputs/{method}/fidelity_time/'
    loop_output_folder = str(ROOT_DIR) + f'/../outputs/{method}/loop/'
    factual_flag = 'fact' if factual else 'cf'
    existed_flag = 'existed' if existed else 'nonexisted'
    sp = int(sparsity*100)
    j_str = str(sp).zfill(3)
    data = f"{config.models.model_name}_{config.datasets.dataset_name}"
    _class = f"{factual_flag}_{existed_flag}"

    for i in range(len(lists)):
        if i % 10 == 0:
            print(i)
        start_time = time.time()
        events = original_events.copy()
        result, num_candidate,initial_output,loop_list,model_time = EA(explainer,model,up_down,factual,lists,events,i,existed,config,pg_index,seed,agents_iteration,flag)
        runtime = time.time()-start_time
        model_time = sum(model_time)
        results.append([result[0],result[1],result[2],runtime,num_candidate,initial_output])
        loop_result = np.array(loop_list,dtype=object)
        if i < 10:
            index = f"0{i}"
        else:
            index = f"{i}"
        loop_file_name = loop_output_folder + f"{_class}/{data}/sp0{sp}_{index}_{factual_flag}_{existed_flag}_{config.models.model_name}_{config.datasets.dataset_name}_loop.csv"
        np.savetxt(loop_file_name, loop_result, fmt='%s', delimiter=',')



    j = int(sparsity * 100)
    j_str = str(j).zfill(3) 
    if factual and existed:
        file_name = output_folder + f"fact_existed/{data}/sp{j_str}_n{n_trials}_{config.models.model_name}_{config.datasets.dataset_name}.csv"
    elif factual and (existed==0):
        file_name = output_folder + f"fact_nonexisted/{data}/sp{j_str}_n{n_trials}_{config.models.model_name}_{config.datasets.dataset_name}.csv"
    elif (factual==0) and existed:
        file_name = output_folder + f"cf_existed/{data}/sp{j_str}_n{n_trials}_{config.models.model_name}_{config.datasets.dataset_name}.csv"
    elif (factual==0) and (existed==0):
        file_name = output_folder + f"cf_nonexisted/{data}/sp{j_str}_n{n_trials}_{config.models.model_name}_{config.datasets.dataset_name}.csv"
    result = np.array(results,dtype=object)
    np.savetxt(file_name, result, fmt='%f', delimiter=',')


if __name__ == '__main__':
    pipeline()


