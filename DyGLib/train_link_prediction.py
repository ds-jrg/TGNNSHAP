import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn

from DyGLib.models.TGAT import TGAT
from DyGLib.models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from DyGLib.models.CAWN import CAWN
from DyGLib.models.TCL import TCL
from DyGLib.models.GraphMixer import GraphMixer
from DyGLib.models.DyGFormer import DyGFormer
from DyGLib.models.modules import MergeLayer, MLPClassifier, NeuralNetworkDst, TGNN, NeuralNetworkSrcDst
from DyGLib.utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer, BatchSubgraphs, NegativeEdgeSampler
from DyGLib.utils.utils import get_neighbor_sampler, compute_stats
from DyGLib.evaluate_models_utils import evaluate_model_link_prediction
from DyGLib.utils.metrics import get_link_prediction_metrics
from DyGLib.utils.DataLoader import get_idx_data_loader, get_link_prediction_data, Data
from DyGLib.utils.EarlyStopping import EarlyStopping

from Config.config import ModelConfig, DataConfig, TrainConfig

def train(modelConfig: ModelConfig, dataConfig: DataConfig, trainConfig: TrainConfig,
          node_raw_features: np.ndarray, edge_raw_features: np.ndarray, 
          full_data: Data, train_data: Data, val_data: Data, test_data: Data):

    warnings.filterwarnings('ignore')

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, edge_features=edge_raw_features, sample_neighbor_strategy=modelConfig.sample_neighbor_strategy,
                                                time_scaling_factor=modelConfig.time_scaling_factor, seed=1)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, edge_features=edge_raw_features, sample_neighbor_strategy=modelConfig.sample_neighbor_strategy,
                                                time_scaling_factor=modelConfig.time_scaling_factor, seed=1)

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
    train_mask = ~np.isnan(train_data.labels) & (train_data.edge_ids != 0)
    val_mask = ~np.isnan(val_data.labels) & (val_data.edge_ids != 0)
    test_mask = ~np.isnan(test_data.labels) & (test_data.edge_ids != 0)
    full_mask = ~np.isnan(full_data.labels) & (full_data.edge_ids != 0)
    train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids[train_mask], dst_node_ids=train_data.dst_node_ids[train_mask], interact_times=train_data.node_interact_times[train_mask], seed=0)
    val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids[full_mask], dst_node_ids=full_data.dst_node_ids[full_mask], interact_times=full_data.node_interact_times[full_mask], seed=0)
    #new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids, dst_node_ids=new_node_val_data.dst_node_ids, seed=1)
    test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids[full_mask], dst_node_ids=full_data.dst_node_ids[full_mask], interact_times=full_data.node_interact_times[full_mask], seed=2)
    #new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3)

    train_idx_data_loader = get_idx_data_loader(indices_list=np.where(train_mask)[0], batch_size=trainConfig.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=np.where(val_mask)[0], batch_size=trainConfig.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=np.where(test_mask)[0], batch_size=trainConfig.batch_size, shuffle=False)

    # get data loaders
    # train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    # val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    # new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    # test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    # new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []
    logger = None

    for run in range(trainConfig.num_runs):

        set_random_seed(seed=run)

        model_name = f'{modelConfig.model_name}_seed{run}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"Slurm/{dataConfig.dataset_name}/{modelConfig.model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"Slurm/{dataConfig.dataset_name}/{modelConfig.model_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {vars(modelConfig)} on {vars(dataConfig)}')

        # create model
        if modelConfig.model_name == 'TGAT':
            dynamic_backbone = TGAT(num_nodes=node_raw_features.shape[0], node_dim=node_raw_features.shape[1], edge_dim=edge_raw_features.shape[1],
                                    time_feat_dim=modelConfig.time_feat_dim, num_layers=modelConfig.num_layers, num_heads=modelConfig.num_heads, dropout=modelConfig.dropout, device=modelConfig.device)
        elif modelConfig.model_name in ['JODIE', 'DyRep', 'TGN']:
            # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
            dynamic_backbone = MemoryModel(num_nodes=node_raw_features.shape[0], node_dim=node_raw_features.shape[1], edge_dim=edge_raw_features.shape[1],
                                           time_feat_dim=modelConfig.time_feat_dim, model_name=modelConfig.model_name, num_layers=modelConfig.num_layers, num_heads=modelConfig.num_heads,
                                           dropout=modelConfig.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=modelConfig.device)
        elif modelConfig.model_name == 'CAWN':
            dynamic_backbone = CAWN(num_nodes=node_raw_features.shape[0], node_dim=node_raw_features.shape[1], edge_dim=edge_raw_features.shape[1],
                                    time_feat_dim=modelConfig.time_feat_dim, position_feat_dim=modelConfig.position_feat_dim, walk_length=modelConfig.walk_length,
                                    num_walk_heads=modelConfig.num_walk_heads, dropout=modelConfig.dropout, device=modelConfig.device)
        elif modelConfig.model_name == 'TCL':
            dynamic_backbone = TCL(num_nodes=node_raw_features.shape[0], node_dim=node_raw_features.shape[1], edge_dim=edge_raw_features.shape[1],
                                    time_feat_dim=modelConfig.time_feat_dim, num_layers=modelConfig.num_layers, num_heads=modelConfig.num_heads,
                                   num_depths=modelConfig.num_neighbors + 1, dropout=modelConfig.dropout, device=modelConfig.device)
        elif modelConfig.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(num_nodes=node_raw_features.shape[0], node_dim=node_raw_features.shape[1], edge_dim=edge_raw_features.shape[1],
                                    time_feat_dim=modelConfig.time_feat_dim, num_tokens=modelConfig.num_neighbors, time_gap=modelConfig.time_gap, num_layers=modelConfig.num_layers, dropout=modelConfig.dropout, device=modelConfig.device)
        elif modelConfig.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(num_nodes=node_raw_features.shape[0], node_dim=node_raw_features.shape[1], edge_dim=edge_raw_features.shape[1],
                                         time_feat_dim=modelConfig.time_feat_dim, channel_embedding_dim=modelConfig.channel_embedding_dim, patch_size=modelConfig.patch_size,
                                         num_layers=modelConfig.num_layers, num_heads=modelConfig.num_heads, dropout=modelConfig.dropout,
                                         max_input_sequence_length=modelConfig.max_input_sequence_length, device=modelConfig.device)
        else:
            raise ValueError(f"Wrong value for model_name {modelConfig.model_name}!")
        
        link_predictor = NeuralNetworkSrcDst(input_dim=node_raw_features.shape[1], num_layers=modelConfig.num_reg_layers, hidden_dim=modelConfig.hidden_reg_layers_dim)
        model = TGNN(dynamic_backbone, link_predictor)
        logger.info(f'model -> {model}')
        logger.info(f'model name: {modelConfig.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        optimizer = create_optimizer(model=model, optimizer_name=trainConfig.optimizer, learning_rate=trainConfig.learning_rate, weight_decay=trainConfig.weight_decay)

        model.to(modelConfig.device)

        save_model_folder = f"Saved_models/{dataConfig.dataset_name}/{modelConfig.model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=trainConfig.patience, save_model_folder=save_model_folder,
                                       save_model_name=model_name, logger=logger, model_name=modelConfig.model_name)

        loss_func = nn.BCELoss()

        for epoch in range(trainConfig.num_epochs):

            model.train()
            if isinstance(model.backbone, MemoryModel):
                # reinitialize memory of memory-based models at the start of each epoch
                model.backbone.memory_bank.__init_memory_bank__()

            # store train losses and metrics
            train_losses, train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                    train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]
                
                batch_edge_features = train_neighbor_sampler.edge_features[batch_edge_ids].to(modelConfig.device)

                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

                subgraphs_src = train_neighbor_sampler.get_multi_hop_neighbors(modelConfig.num_layers, batch_src_node_ids, batch_node_interact_times, num_neighbors = modelConfig.num_neighbors)
                subgraphs_dst = train_neighbor_sampler.get_multi_hop_neighbors(modelConfig.num_layers, batch_dst_node_ids, batch_node_interact_times, num_neighbors = modelConfig.num_neighbors)
                subgraphs_dst_neg = train_neighbor_sampler.get_multi_hop_neighbors(modelConfig.num_layers, batch_neg_dst_node_ids, batch_node_interact_times, num_neighbors = modelConfig.num_neighbors)
                edge_feat_src = train_neighbor_sampler.get_edge_features_for_multi_hop(subgraphs_src[1])
                edge_feat_dst = train_neighbor_sampler.get_edge_features_for_multi_hop(subgraphs_dst[1])
                edge_feat_dst_neg = train_neighbor_sampler.get_edge_features_for_multi_hop(subgraphs_dst_neg[1])

                subgraphs_src = BatchSubgraphs(*subgraphs_src, edge_feat_src)
                subgraphs_src.to(modelConfig.device)
                subgraphs_dst = BatchSubgraphs(*subgraphs_dst, edge_feat_dst)
                subgraphs_dst.to(modelConfig.device)
                subgraphs_dst_neg = BatchSubgraphs(*subgraphs_dst_neg, edge_feat_dst_neg)
                subgraphs_dst_neg.to(modelConfig.device)

                negative_probabilities = model(src_node_ids=batch_src_node_ids,
                                dst_node_ids=batch_neg_dst_node_ids,
                                node_interact_times=batch_node_interact_times,
                                src_subgraphs = subgraphs_src,
                                dst_subgraphs = subgraphs_dst_neg,
                                num_neighbors=modelConfig.num_neighbors,
                                edge_features=None,
                                edges_are_positive=False).squeeze(dim=-1).sigmoid()

                positive_probabilities = model(src_node_ids=batch_src_node_ids,
                                dst_node_ids=batch_dst_node_ids,
                                node_interact_times=batch_node_interact_times,
                                src_subgraphs = subgraphs_src,
                                dst_subgraphs = subgraphs_dst,
                                num_neighbors=modelConfig.num_neighbors,
                                edge_features=batch_edge_features,
                                edges_are_positive=True).squeeze(dim=-1).sigmoid()
                

                predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

                loss = loss_func(input=predicts, target=labels)

                train_losses.append(loss.item())

                train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')

                if isinstance(model.backbone, MemoryModel):
                    # detach the memories and raw messages of nodes in the memory bank after each batch, so we don't back propagate to the start of time
                    model.backbone.memory_bank.detach_memory_bank()

            if isinstance(model.backbone, MemoryModel):
                # backup memory bank after training so it can be used for new validation nodes
                train_backup_memory_bank = model.backbone.memory_bank.backup_memory_bank()

            val_losses, val_metrics = evaluate_model_link_prediction(model=model,
                                                                    neighbor_sampler=full_neighbor_sampler,
                                                                    evaluate_idx_data_loader=val_idx_data_loader,
                                                                    evaluate_data=val_data,
                                                                    evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                    loss_func=loss_func,
                                                                    num_neighbors=modelConfig.num_neighbors,
                                                                    num_layers=modelConfig.num_layers,
                                                                    time_gap=modelConfig.time_gap,
                                                                    device=modelConfig.device)
            if isinstance(model.backbone, MemoryModel):
                # backup memory bank after validating so it can be used for testing nodes (since test edges are strictly later in time than validation edges)
                val_backup_memory_bank = model.backbone.memory_bank.backup_memory_bank()

                # reload training memory bank for new validation nodes
                model.backbone.memory_bank.reload_memory_bank(train_backup_memory_bank)

            #Skip evaluation on new nodes

            if isinstance(model.backbone, MemoryModel):
                # reload validation memory bank for testing nodes or saving models
                # note that since model treats memory as parameters, we need to reload the memory to val_backup_memory_bank for saving models
                model.backbone.memory_bank.reload_memory_bank(val_backup_memory_bank)

            logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
            for metric_name in train_metrics[0].keys():
                logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                logger.info(f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')

            # perform testing once after test_interval_epochs
            if (epoch + 1) % trainConfig.test_interval_epochs == 0:
                test_losses, test_metrics = evaluate_model_link_prediction(model=model,
                                                                    neighbor_sampler=full_neighbor_sampler,
                                                                    evaluate_idx_data_loader=test_idx_data_loader,
                                                                    evaluate_data=test_data,
                                                                    evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                    loss_func=loss_func,
                                                                    num_neighbors=modelConfig.num_neighbors,
                                                                    num_layers=modelConfig.num_layers,
                                                                    time_gap=modelConfig.time_gap,
                                                                    device=modelConfig.device)

                if isinstance(model.backbone, MemoryModel):
                    # reload validation memory bank for testing nodes or saving models
                    # note that since model treats memory as parameters, we need to reload the memory to val_backup_memory_bank for saving models
                    model.backbone.memory_bank.reload_memory_bank(val_backup_memory_bank)

                logger.info(f'test loss: {np.mean(test_losses):.4f}')
                for metric_name in test_metrics[0].keys():
                    logger.info(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')

            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics[0].keys():
                val_metric_indicator.append((metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
            intermediate_val_metrics = {
                metric_name: np.mean([val_metric[metric_name] for val_metric in val_metrics])
                for metric_name in val_metrics[0].keys()
            }
            early_stop = early_stopping.step(
                val_metric_indicator,
                model,
                intermediate_val_metrics,
                save_model_folder,
                model_name,
            )

            if early_stop:
                break

        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f'get final performance on dataset {dataConfig.dataset_name}...')

        # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
        val_losses, val_metrics = np.array([]), np.array([])
        if not isinstance(model.backbone, MemoryModel):
            val_losses, val_metrics = evaluate_model_link_prediction(model=model,
                                                                    neighbor_sampler=full_neighbor_sampler,
                                                                    evaluate_idx_data_loader=val_idx_data_loader,
                                                                    evaluate_data=val_data,
                                                                    evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                    loss_func=loss_func,
                                                                    num_neighbors=modelConfig.num_neighbors,
                                                                    num_layers=modelConfig.num_layers,
                                                                    time_gap=modelConfig.time_gap,
                                                                    device=modelConfig.device)

        test_losses, test_metrics = evaluate_model_link_prediction(model=model,
                                                                    neighbor_sampler=full_neighbor_sampler,
                                                                    evaluate_idx_data_loader=test_idx_data_loader,
                                                                    evaluate_data=test_data,
                                                                    evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                    loss_func=loss_func,
                                                                    num_neighbors=modelConfig.num_neighbors,
                                                                    num_layers=modelConfig.num_layers,
                                                                    time_gap=modelConfig.time_gap,
                                                                    device=modelConfig.device)

        
        # store the evaluation metrics at the current run
        val_metric_dict, new_node_val_metric_dict, test_metric_dict, new_node_test_metric_dict = {}, {}, {}, {}

        if not isinstance(model.backbone, MemoryModel):
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
                logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
                val_metric_dict[metric_name] = average_val_metric


        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        if not isinstance(model.backbone, MemoryModel):
            val_metric_all_runs.append(val_metric_dict)
            new_node_val_metric_all_runs.append(new_node_val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)
        new_node_test_metric_all_runs.append(new_node_test_metric_dict)

        # avoid the overlap of logs
        if run < trainConfig.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        if not isinstance(model.backbone, MemoryModel):
            result_json = {
                "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
                "new node validate metrics": {metric_name: f'{new_node_val_metric_dict[metric_name]:.4f}' for metric_name in new_node_val_metric_dict},
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
                "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
            }
        else:
            result_json = {
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
                "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
            }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = save_model_folder
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)
    if logger is not None:
        # store the average metrics at the log of the last run
        logger.info(f'metrics over {trainConfig.num_runs} runs:')

        if modelConfig.model_name not in ['JODIE', 'DyRep', 'TGN']:
            for metric_name in val_metric_all_runs[0].keys():
                logger.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
                logger.info(f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                            f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

        for metric_name in test_metric_all_runs[0].keys():
            logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
            logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                        f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    return val_metric_all_runs, test_metric_all_runs
