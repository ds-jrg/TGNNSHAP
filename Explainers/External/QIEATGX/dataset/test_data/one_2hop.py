import torch
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import multiprocessing as mp
from multiprocessing import Process
import numpy as np
import pandas as pd
import itertools

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

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

    # save originals
    tgt_src1 = events.iloc[tgt_index-1,0]
    tgt_dst1 = events.iloc[tgt_index-1,1]
    tgt_time1 = events.iloc[tgt_index-1,2]
    
    if existed == 1:
        assert tgt_src1 == tgt_src
        assert tgt_dst1 == tgt_dst
        assert tgt_time1 == tgt_time
    
    return tgt_src,tgt_dst,tgt_time,tgt_index,tgt_src1,tgt_dst1,tgt_time1


def get_hop_run(events0,model,lists,j,start_time,factual,existed,sparsity,config,explainer,one_loop):
    
    if torch.cuda.is_available() and config.explainers.use_gpu:
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')


    len_events = len(events0)
    tgt_src,tgt_dst,tgt_time,tgt_index,tgt_src1,tgt_dst1,tgt_time1 = set_data(factual,lists,events0,j,existed)
    
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

    accu_e_idx = [ ] # NOTE: important?
    accu_node = [tgt_src, tgt_dst,]
    accu_ts = [tgt_time, tgt_time,]
    ngh_finder = construct_tgat_neighbor_finder(events)
    # num_neighbors = 2
    hop_events = []
    num_neighbors = int(config.models.param.num_neighbors)
    
    out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = ngh_finder.get_temporal_neighbor(
                                                                        accu_node, 
                                                                        accu_ts, 
                                                                        num_neighbors=num_neighbors,
                                                                        )
    out_ngh_eidx_batch = out_ngh_eidx_batch.flatten() 
    out_ngh_t_batch = out_ngh_t_batch.flatten()
    out_ngh_node_batch = out_ngh_node_batch.flatten()

    out_ngh_eidx_batch = [event for event in out_ngh_eidx_batch if event != 0]
    out_ngh_t_batch = [event for event in out_ngh_t_batch if event != 0]
    out_ngh_node_batch = [event for event in out_ngh_node_batch if event != 0]
    accu_node = out_ngh_node_batch
    accu_ts = out_ngh_t_batch
    out_ngh_eidx_batch = out_ngh_eidx_batch
    onehop_events = out_ngh_eidx_batch

    assert len(accu_node) == len(onehop_events)
    assert len(accu_node) == len(accu_ts)

    out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = ngh_finder.get_temporal_neighbor(
                                                                        accu_node, 
                                                                        accu_ts, 
                                                                        num_neighbors=num_neighbors,
                                                                        edge_idx_preserve_list=events.index, # NOTE: not needed?
                                                                        )
    assert len(accu_node) == len(out_ngh_eidx_batch)
    out_ngh_eidx_batch2 = out_ngh_eidx_batch.flatten()
    twohop_events = []
    for i in range(len(out_ngh_eidx_batch)):
        a = out_ngh_eidx_batch[i].tolist()
        if type(a) == int:
            a = [a]
        events_2hop = [event for event in a if event != 0]
        twohop_events.append(events_2hop)
        

    accu_e_idx.append(onehop_events)
    accu_e_idx.append(out_ngh_eidx_batch2)
    assert len(onehop_events) == len(twohop_events)

    unique_e_idx = np.array(list(itertools.chain.from_iterable(accu_e_idx)))
    unique_e_idx = unique_e_idx[ unique_e_idx != 0 ] # NOTE: 0 are padded e_idxs
    unique_e_idx = np.unique(unique_e_idx).tolist()

    assert unique_e_idx == candiates_2hop
    
    result = [ [] for _ in range(len(onehop_events))]
    for i in range(len(onehop_events)):
        result[i].append(onehop_events[i])

        for j in range(len(out_ngh_eidx_batch[i])):
            result[i].append(out_ngh_eidx_batch[i][j])

    first_row_length = len(result[0])
    assert all(len(row) == first_row_length for row in result)

    
    return result
    

@hydra.main(config_path='../../Xmethods/config', config_name='config', version_base=None)
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
    
    # print(OmegaConf.to_yaml(config))

    # import ipdb; ipdb.set_trace()

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


    hops_holder =  str(ROOT_DIR) + f'/../dataset/hops/'



    factual_flag = 'fac' if factual else 'cf'
    existed_flag = 'existed' if existed else 'nonexisted'
    sp = int(sparsity*100)
    data = f"{config.models.model_name}_{config.datasets.dataset_name}"
    _class = f"{factual_flag}_{existed_flag}"
    if config.datasets.dataset_name == 'wikipedia':
        data = f"{config.models.model_name}_wiki"
    else:
        data = f"{config.models.model_name}_redi"
    one_loop = 0
    for j in range(len(lists)):

        if j%10 == 1:
            print(j*10)
        start_time = time.time()
        events0 = events.copy()
        result = get_hop_run(events0,model,lists,j,start_time,factual,existed,sparsity,config,explainer,one_loop)
        

        hop_result = np.array(result,dtype=object)
        if j < 10:
            index = f"0{j}"
        else:
            index = f"{j}"

        file_name = hops_holder + f"{_class}/{data}/i{index}_{factual_flag}_{existed_flag}_{config.models.model_name}_{config.datasets.dataset_name}.csv"
        np.savetxt(file_name, hop_result, fmt='%d', delimiter=',') 

    

if __name__ == '__main__':
    pipeline()


