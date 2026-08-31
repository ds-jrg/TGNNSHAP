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
        


def one_process(config,explainer,lists,events,model,sparsity,i,factual,existed,pg_explainer_model):
    tgt_src,tgt_dst,tgt_time,tgt_index,tgt_src1,tgt_dst1,tgt_time1 = set_data(factual,lists,events,i,existed)
    print(tgt_src,tgt_dst,tgt_time,tgt_index)
    if existed == 0:
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

    if factual == 0:
        if existed == 0:
            assert initial_output < 0.5


    from Xmethods.codes.tgnnexplainer.method.subgraphx_tg_for_cf2 import SubgraphXTG
    device = torch.device('cuda', index=config.device_id)
    n_rollout = int(config.n_trials)
    # print('used pg_explainer_tg ckpt:', explainer_ckpt_path)
    tgt_index = [tgt_index,]

    explainer = SubgraphXTG(sparsity,
                            model, 
                            # lists,
                            config.models.model_name, 
                            config.explainers.explainer_name,
                            config.datasets.dataset_name,
                            events,
                            config.explainers.param.explanation_level, 
                            device,
                            results_dir=config.explainers.results_dir,
                            debug_mode=config.explainers.debug_mode,
                            save_results=config.explainers.results_save,
                            mcts_saved_dir=config.explainers.mcts_saved_dir,
                            load_results=config.explainers.load_results,
                            rollout=n_rollout,
                            min_atoms=config.explainers.param.min_atoms,
                            c_puct=config.explainers.param.c_puct,
                            pg_explainer_model=pg_explainer_model if config.explainers.use_pg_explainer else None,
                            pg_positive=config.explainers.pg_positive,
                            factual=factual,
                            existed=existed,
                            )

        # ここから推論
    explain_results,end_rollout,result_list,loop_result = explainer(event_idxs=tgt_index, sparsity = sparsity)

    # 評価を行う2
    from Xmethods.codes.tgnnexplainer.evaluation.metrics_tg2_cf import EvaluatorMCTSTG
    evaluator = EvaluatorMCTSTG(model_name=config.models.model_name,
                                explainer_name=config.explainers.explainer_name,
                                dataset_name=config.datasets.dataset_name,
                                explainer=explainer[0] if isinstance(explainer, list) else explainer,
                                results_dir=config.explainers.results_dir
                                ) 

    if config.evaluate:
        # evaluator.evaluate(explain_results, event_idxs=tgt_index)
        flag,fidelity,remove_use_times,num_2hop,best_output= evaluator.evaluate(explain_results, event_idxs=tgt_index, ori_cfpred=initial_output,factual=factual,existed=existed)
    else:
        AssertionError('no evaluate.')

    for k in range(len(loop_result)):
        loop_result[k][2]=initial_output
    
    for k in range(len(result_list)):
        result_list[k][2]=initial_output
    
    return flag,fidelity,remove_use_times,num_2hop,best_output,initial_output,end_rollout,result_list,loop_result



@hydra.main(config_path='../config', config_name='config', version_base=None)
def pipeline(config: DictConfig):
    # model config
    config.models.param = config.models.param[config.datasets.dataset_name]
    config.models.ckpt_path = str(ROOT_DIR/'xgraph'/'models'/'checkpoints'/f'{config.models.model_name}_{config.datasets.dataset_name}_best.pth')

    # dataset config
    config.datasets.dataset_path = str(ROOT_DIR/'..'/'dataset'/'data'/f'{config.datasets.dataset_name}.csv')
    config.datasets.explain_idx_filepath = str(ROOT_DIR/'..'/'dataset'/'explain_index'/f'{config.datasets.explain_idx_filename}.csv')

    # explainer config
    config.explainers.param = config.explainers.param[config.datasets.dataset_name]
    config.explainers.results_dir = str(ROOT_DIR.parent/'Xmethods'/'codes'/'tgnnexplainer'/'results')
    config.explainers.mcts_saved_dir = str(ROOT_DIR.parent/'Xmethods'/'codes'/'tgnnexplainer'/'saved_mcts_results')
    config.explainers.explainer_ckpt_dir = str(ROOT_DIR/'xgraph'/'explainer_ckpts')

    # import ipdb; ipdb.set_trace()
    if torch.cuda.is_available() and config.explainers.use_gpu:
        device = torch.device('cuda', index=config.device_id)
    else:
        AssertionError('no gpu.')
        device = torch.device('cpu')
    device = torch.device('cuda', index=config.device_id)
    # DONE: only use tgat processed data
    events, edge_feats, node_feats = load_tg_dataset(config.datasets.dataset_name)
    target_event_idxs = load_explain_idx(config.datasets.explain_idx_filepath, start=0)
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


    sparsity = float(config.sparse_ratio)
    existed = int(config.existed)
    factual = int(config.factual)
    if (existed == 1) and (factual == 1):
        file_name = str(ROOT_DIR) + f"/../dataset/test_data/test_existed_fac_{config.models.model_name}_{config.datasets.dataset_name}.csv"
    elif (existed == 1) and (factual == 0):
        file_name = str(ROOT_DIR) + f"/../dataset/xgraph/test_data/test_existed_cf_{config.models.model_name}_{config.datasets.dataset_name}.csv"
    elif (existed == 0) and (factual == 1):
        file_name = str(ROOT_DIR) + f"/../dataset/xgraph/test_data/test_nonexisted_fac_{config.models.model_name}_{config.datasets.dataset_name}.csv"
    elif (existed == 0) and (factual == 0):
        file_name = str(ROOT_DIR) + f"/../dataset/xgraph/test_data/test_nonexisted_cf_{config.models.model_name}_{config.datasets.dataset_name}.csv"
    lists = np.loadtxt(file_name, delimiter=',')
    if config.explainers.explainer_name == 'subgraphx_tg': # DONE: test this 'use_pg_explainer'
        from Xmethods.codes.tgnnexplainer.method.other_baselines_tg import PGExplainerExt
        pg_explainer_model, explainer_ckpt_path = PGExplainerExt.expose_explainer_model(model, # load a trained mlp model
                                model_name=config.models.model_name,
                                explainer_name='pg_explainer_tg', # fixed
                                dataset_name=config.datasets.dataset_name,
                                ckpt_dir=config.explainers.explainer_ckpt_dir,
                                device=device,
                                )

        from Xmethods.codes.tgnnexplainer.method.subgraphx_tg_for_cf import SubgraphXTG
        factual = int(config.factual)
        n_rollout = int(config.n_trials)

        explainer = SubgraphXTG(sparsity,
                                model, 
                                # lists,
                                config.models.model_name, 
                                config.explainers.explainer_name,
                                config.datasets.dataset_name,
                                events,
                                config.explainers.param.explanation_level, 
                                device,
                                results_dir=config.explainers.results_dir,
                                debug_mode=config.explainers.debug_mode,
                                save_results=config.explainers.results_save,
                                mcts_saved_dir=config.explainers.mcts_saved_dir,
                                load_results=config.explainers.load_results,
                                rollout=n_rollout,
                                min_atoms=config.explainers.param.min_atoms,
                                c_puct=config.explainers.param.c_puct,
                                pg_explainer_model=pg_explainer_model if config.explainers.use_pg_explainer else None,
                                pg_positive=config.explainers.pg_positive,
                                factual=factual,
                                existed=existed,
                                )


       

    # save folder
    method = f'T-GNNExplainer'
    output_folder = str(ROOT_DIR) + f'/../outputs/{method}/fidelity_time/'
    loop_output_folder = str(ROOT_DIR) + f'/../outputs/{method}/loop/'
    
    factual_flag = 'fact' if factual else 'cf'
    existed_flag = 'existed' if existed else 'nonexisted'
    sp = int(sparsity*100)
    data = f"{config.models.model_name}_{config.datasets.dataset_name}"
    _class = f"{factual_flag}_{existed_flag}"

    max_time = 600 # fix

    # run the explainer
    values = []
    result_list = [[] for _ in range(int(sparsity*10))]
    random_result = []
    for i in range(len(lists)):
        if i == 1:
            break
        # i = 0
        if i%10 == 1:
            print(i*10)

        if i < 10:
            index = f"0{i}"
        else:
            index = f"{i}"
        original_events = events.copy()
        start_time = time.time()
        flag,fidelity,remove_use_times,num_2hop,best_output,initial_output,end_rollout,result_list,loop_result = one_process(config,explainer,lists,events,model,sparsity,i,factual,existed,pg_explainer_model)

        one_result = np.array(result_list,dtype=object)
        loop_result = np.array(loop_result,dtype=object)
        runtime = time.time() - start_time

        loop_file_name = loop_output_folder + f"{_class}/{data}/sp0{sp}_i{index}_{factual_flag}_{existed_flag}_{config.models.model_name}_{config.datasets.dataset_name}_loop.csv"
        np.savetxt(loop_file_name, loop_result, fmt='%s', delimiter=',')
        r = 0
        if runtime > 600:
            r = 1
        random_result.append([flag,fidelity,remove_use_times,runtime,num_2hop,best_output,initial_output,end_rollout,r])

    j = int(sparsity*10)
    if factual and existed:
        file_name = output_folder + f"fact_existed/{data}/base0_{j}_r{n_rollout}_factual_pg_fidelity_{sparsity}_{config.models.model_name}_{config.datasets.dataset_name}.csv"
    elif factual and (existed==0):
        file_name = output_folder + f"fact_nonexisted/{data}/base0_{j}_r{n_rollout}_factual_nonexisted_pg_fidelity_{sparsity}_{config.models.model_name}_{config.datasets.dataset_name}.csv"
    elif (factual==0) and existed:
        file_name = output_folder + f"cf_existed/{data}/base0_{j}_r{n_rollout}_cf_pg_fidelity_{sparsity}_{config.models.model_name}_{config.datasets.dataset_name}.csv"
    elif (factual==0) and (existed==0):
        file_name = output_folder + f"cf_nonexisted/{data}/base0_{j}_r{n_rollout}_cf_nonexisted_pg_fidelity_{sparsity}_{config.models.model_name}_{config.datasets.dataset_name}.csv"
    result = np.array(random_result,dtype=object)
    np.savetxt(file_name, result, fmt='%f', delimiter=',')





if __name__ == '__main__':
    pipeline()


