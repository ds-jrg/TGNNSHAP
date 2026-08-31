import torch
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import multiprocessing as mp
from multiprocessing import Process

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from dataset.tg_dataset import load_tg_dataset, load_explain_idx
from dataset.utils_dataset import construct_tgat_neighbor_finder

from TGNNmodels.xgraph.models.ext.tgat.module import TGAN
from TGNNmodels.xgraph.models.ext.tgn.model.tgn import TGN
from TGNNmodels.xgraph.models.ext.tgn.utils.data_processing import compute_time_statistics
from TGNNmodels import ROOT_DIR

import numpy as np


def set_data(factual,lists,events,i,existed):
    tgt_src = int(lists[i][0])
    tgt_dst = int(lists[i][1])
    tgt_time = float(lists[i][2])
    tgt_index = int(lists[i][3])

    tgt_src1 = events.iloc[tgt_index-1,0]
    tgt_dst1 = events.iloc[tgt_index-1,1]
    tgt_time1 = events.iloc[tgt_index-1,2]
    
    if existed == 1:
        assert tgt_src1 == tgt_src
        assert tgt_dst1 == tgt_dst
        assert tgt_time1 == tgt_time
    
    return tgt_src,tgt_dst,tgt_time,tgt_index,tgt_src1,tgt_dst1,tgt_time1


def pg_run(events0,model,lists,j,factual,existed,sparsity,config,explainer,start_time):
    start_time = time.time()
    if torch.cuda.is_available() and config.explainers.use_gpu:
        device = torch.device('cuda', index=config.device_id)
    else: 
        AssertionError('GPU only')

    tgt_src,tgt_dst,tgt_time,tgt_index,tgt_src1,tgt_dst1,tgt_time1 = set_data(factual,lists,events0,j,existed)
    events = events0.copy()

    if existed==0:
        events.iloc[tgt_index-1,0] = tgt_src
        events.iloc[tgt_index-1,1] = tgt_dst
        events.iloc[tgt_index-1,2] = tgt_time
    src_idx_l = np.array([tgt_src, ])
    target_idx_l = np.array([tgt_dst, ])
    cut_time_l = np.array([tgt_time, ])
    input_data = [src_idx_l, target_idx_l, cut_time_l]
    output1 = model.get_prob( *input_data,  edge_idx_preserve_list=events)
    ori_ori_value = output1.cpu().detach().numpy()
    initial_output = ori_ori_value[0]
    output = model.get_prob( *input_data,  edge_idx_preserve_list=events)
    no_use_value = output.cpu().detach().numpy()
    no_use_output = no_use_value[0]
    
    if existed == 0: 
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

    tgt_index = [tgt_index,]
    results_list,candiates= explainer(event_idxs=tgt_index,tgt_src=tgt_src,tgt_dst=tgt_dst,sparsity=sparsity,existed=existed,factual=factual)

    candidate_sorted = results_list[-1][0]
    
    len_candidate = len(candidate_sorted)
    result = []
    sparsity_list = []
    return_list = []
    for i in range(1, int(sparsity*10+1)):
        sparsity_list.append(max(1, int(round(len_candidate * (i / 10)))))

    # print('2hop:',len_candidate,'use:',int(len_candidate*sparsity))
    if (factual == existed):
        best_output = 0
    else:
        best_output = 1

    result_list = []
    loop_list = []
    for i in range(int(len(candidate_sorted))):
        a = i+1
        if factual:
            if existed:
                use_events = candidate_sorted[:a] 
            else:
                use_events = candidate_sorted[len_candidate-a:] 
            
        else:
            if existed:
                use_events = candidate_sorted[a:] 
            else:
                use_events = candidate_sorted[:len_candidate-a]  

        output2 = model.get_prob( src_idx_l, target_idx_l, cut_time_l, edge_idx_preserve_list=use_events)
        output2 = output2.cpu().detach().numpy()[0]
        result.append(output2)
        runtime = time.time()-start_time
        result_list.append([output2,a,initial_output,runtime])
        candiates_str = ','.join(map(str, use_events))
        loop_list.append([output2,a,candiates_str])
        
        if (i+1) in sparsity_list: 
            flag = 0
            values = np.array(result)
            if (factual != existed): 
                use_rm_events = np.argmin(values)
                best_output = np.min(values)
                if best_output < 0.5:
                    flag = 1
                fidelity = initial_output-best_output
            else: 
                use_rm_events = np.argmax(values)  
                best_output = np.max(values)
                if best_output > 0.5:
                    flag = 1
                fidelity = best_output-initial_output
            one_time = time.time()-start_time
            return_list.append([flag,fidelity,use_rm_events+1,one_time])
            if (i+1) == sparsity_list[-1]:
                break
    
    return return_list, len_candidate, initial_output,result_list,loop_list
            


@hydra.main(config_path='../config', config_name='config', version_base=None)
def pipeline(config: DictConfig):
    # model config
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
        AssertionError('GPU only')

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

    state_dict = torch.load(config.models.ckpt_path, weights_only=True)
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
    method = f'PGExplainer'
    output_folder = str(ROOT_DIR) + f'/../outputs/{method}/fidelity_time/'
    loop_output_folder = str(ROOT_DIR) + f'/../outputs/{method}/loop/'
    
    factual_flag = 'fact' if factual else 'cf'
    existed_flag = 'existed' if existed else 'nonexisted'
    data = f"{config.models.model_name}_{config.datasets.dataset_name}"
    _class = f"{factual_flag}_{existed_flag}"

    original_events = events.copy()
    values = []
    results = []
    result_list = []
    sparsity = float(config.sparse_ratio)
    sparsity_list = []
    for i in range(int(sparsity*10)):
        result_list.append([])
    for i in range(len(lists)):
        # if i == 1:
        #     break
        if i%10 == 1:
            print(i*10)
        if i < 10:
            index = f"0{i}"
        else:
            index = f"{i}"
        start_time = time.time()
        events = original_events.copy()

        result, num_2hop, initial_output,result_list2,loop_list = pg_run(events,model,lists,i,factual,existed,sparsity,config,explainer,start_time)
        one_result = np.array(result_list2,dtype=object)
        loop_result = np.array(loop_list,dtype=object)
        for j in range(len(result)):
            sp = int(j*100)
            result_list[j].append([result[j][0],result[j][1],result[j][2],result[j][3],num_2hop, initial_output])

        loop_file_name = loop_output_folder + f"{_class}/{data}/_{index}_{factual_flag}_{existed_flag}_{config.models.model_name}_{config.datasets.dataset_name}_loop.csv"
        np.savetxt(loop_file_name, loop_result, fmt='%s', delimiter=',')

    for i in range(int(sparsity*10)):
        j = i+1
        j = int(j* 10)
        j_str = str(j).zfill(3)    
        if factual and existed:
            file_name = output_folder + f"fact_existed/{data}/base3_sp{j_str}_factual_pg_fidelity_{sparsity}_{config.models.model_name}_{config.datasets.dataset_name}.csv"
        elif factual and (existed==0):
            file_name = output_folder + f"fact_nonexisted/{data}/base3_sp{j_str}_factual_nonexisted_pg_fidelity_{sparsity}_{config.models.model_name}_{config.datasets.dataset_name}.csv"
        elif (factual==0) and existed:
            file_name = output_folder + f"cf_existed/{data}/base3_sp{j_str}_cf_pg_fidelity_{sparsity}_{config.models.model_name}_{config.datasets.dataset_name}.csv"
        elif (factual==0) and (existed==0):
            file_name = output_folder + f"cf_nonexisted/{data}/base3_sp{j_str}_cf_nonexisted_pg_fidelity_{sparsity}_{config.models.model_name}_{config.datasets.dataset_name}.csv"
    
        result = np.array(result_list[i],dtype=object)
        np.savetxt(file_name, result, fmt='%f', delimiter=',')
    print('finish')


if __name__ == '__main__':
    pipeline()


