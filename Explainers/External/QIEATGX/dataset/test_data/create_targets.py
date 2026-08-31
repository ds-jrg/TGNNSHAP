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





@hydra.main(config_path='../../Xmethods/config', config_name='config', version_base=None)
def pipeline(config: DictConfig):
    # model config
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
        device = torch.device('cpu')

    # DONE: only use tgat processed data
    events, edge_feats, node_feats = load_tg_dataset(config.datasets.dataset_name)
    ngh_finder = construct_tgat_neighbor_finder(events)


    existed = int(config.existed)
    factual = int(config.factual)

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


    size_target = 100
    list = []

    ori_events = events.copy()

    for i in range(size_target):
        start_time = time.time()
        if i%10 == 1:
            print(i*10)
        tgt_can_lists = []
        while True:
            time1 = time.time()
            
            random_seed = int(time.time())
            random_index = events[events.index >= 10000].sample(n=1, random_state=random_seed).index[0]

            if existed == 1:
                tgt_src = events.iloc[random_index-1,0]
                tgt_dst = events.iloc[random_index-1,1]
                tgt_time = events.iloc[random_index-1,2]
            else:
                random_index = events[events.index >= 10000].sample(n=1, random_state=random_seed).index[0]
                tgt_src = events.iloc[random_index-1,0]
                random_index = events[events.index >= 10000].sample(n=1, random_state=random_seed).index[0]
                tgt_dst = events.iloc[random_index-1,1]
                random_index = events[events.index >= 10000].sample(n=1, random_state=random_seed).index[0]
                tgt_time = events.iloc[random_index-1,2]

            events0 = ori_events.copy()
            tgt_index = random_index
            if existed==0:
                events0 = ori_events.copy()
                events0.iloc[tgt_index-1,0] = tgt_src
                events0.iloc[tgt_index-1,1] = tgt_dst
                events0.iloc[tgt_index-1,2] = tgt_time
                explainer = PGExplainerExt(
                                model,
                                config.models.model_name,
                                config.explainers.explainer_name,
                                config.datasets.dataset_name,
                                events0,
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
            candiate_2hop = explainer.candidate_events
            if len(candiate_2hop) == 0:
                continue
            src_idx_l = np.array([tgt_src, ])
            target_idx_l = np.array([tgt_dst, ])
            cut_time_l = np.array([tgt_time, ])
            
            input_data = [src_idx_l, target_idx_l, cut_time_l]

            output = explainer.model.get_prob( *input_data, edge_idx_preserve_list=events)
            # output2 = explainer.model.get_prob( *input_data2, edge_idx_preserve_list=events)
            assert output > 0.0
            assert output < 1.0
            # print(output)
            _list = [tgt_src, tgt_dst, tgt_time,random_index]
            if _list in list:
                continue
            
            if (existed == 1):
                if output > 0.5:
                    break
                else:
                    continue
            else:
                if output < 0.5:
                    break
                else:
                    continue

            


        if len(list) == 100:
            break
        # print(_list)
        list.append(_list)

        

    output_folder = str(ROOT_DIR) + f'/../dataset/test_data'
    result = np.array(list[:100])

    if (existed == 1) and (factual == 1):
        file_name = output_folder + f"/new_test_existed_fac_{config.models.model_name}_{config.datasets.dataset_name}.csv"
    elif (existed == 1) and (factual == 0):
        file_name = output_folder + f"/new_test_existed_cf_{config.models.model_name}_{config.datasets.dataset_name}.csv"
    elif (existed == 0) and (factual == 1):
        file_name = output_folder + f"/new_test_data/test_nonexisted_fac_{config.models.model_name}_{config.datasets.dataset_name}.csv"
    elif (existed == 0) and (factual == 0):
        file_name = output_folder + f"/new_test_nonexisted_cf_{config.models.model_name}_{config.datasets.dataset_name}.csv"
    np.savetxt(file_name, result, fmt='%f', delimiter=',')


if __name__ == '__main__':
    pipeline()


