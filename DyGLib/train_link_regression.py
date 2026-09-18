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
from DyGLib.models.modules import MergeLayer, MLPClassifier, NeuralNetworkSrcDst, TGNN
from DyGLib.utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer, BatchSubgraphs
from DyGLib.utils.utils import get_neighbor_sampler, compute_stats, NeighborSampler
from DyGLib.evaluate_models_utils import evaluate_model_link_regression
from DyGLib.utils.metrics import get_link_regression_metrics
from DyGLib.utils.DataLoader import get_idx_data_loader, Data
from DyGLib.utils.EarlyStopping import EarlyStopping

from Config.config import ModelConfig, DataConfig, TrainConfig

def train(modelConfig: ModelConfig, dataConfig: DataConfig, trainConfig: TrainConfig,
          node_raw_features: np.ndarray, edge_raw_features: np.ndarray, 
          full_data: Data, train_data: Data, val_data: Data, test_data: Data):
    """
    Train, validate, and evaluate a temporal graph neural network (TGNN) model on a link regression task.

    This function supports a variety of dynamic GNN backbones including:
    TGAT, JODIE, DyRep, TGN, CAWN, TCL, GraphMixer, DyGFormer.

    Workflow:
    ----------
    1. Initialize graph neighbor samplers
    2. For each run (random seed):
        - Create logger
        - Build model & optimizer
        - Execute epoch training loop:
            a. Sample batches from train data
            b. Build multi-hop temporal subgraphs
            c. Forward pass, compute loss, and backprop
            d. Evaluate on validation and possibly test
        - Apply EarlyStopping
        - Save best model & results
    3. Aggregate results across runs, log mean & std metrics.

    Parameters
    ----------
    modelConfig : ModelConfig
        Model hyperparameters & architecture config.
    dataConfig : DataConfig
        Dataset configuration.
    trainConfig : TrainConfig
        Training configuration.
    node_raw_features : np.ndarray
        Node feature matrix [num_nodes, num_features].
    edge_raw_features : np.ndarray
        Edge feature matrix [num_edges, num_features].
    full_data : Data
        The full temporal graph dataset.
    train_data : Data
        Training split.
    val_data : Data
        Validation split.
    test_data : Data
        Test split.

    Returns
    -------
    val_metric_all_runs : list
        Validation metrics for each run.
    test_metric_all_runs : list
        Test metrics for each run.
    """

    # Neighbor sampler for retrieving temporal subgraphs for every batch
    full_neighbor_sampler = get_neighbor_sampler(
        data=full_data,
        edge_features=edge_raw_features,
        sample_neighbor_strategy=modelConfig.sample_neighbor_strategy,
        time_scaling_factor=modelConfig.time_scaling_factor,
        seed=1
    )

    # Store all validation & test results for each run
    val_metric_all_runs, test_metric_all_runs = [], []

    # ---------------------
    # MULTIPLE SEED RUNS
    # ---------------------
    for run in range(trainConfig.num_runs):
        set_random_seed(seed=run)
        model_name = f'{modelConfig.model_name}_seed{run}'

        # Logging setup for each run
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"Slurm/{dataConfig.dataset_name}/{modelConfig.model_name}/", exist_ok=True)

        fh = logging.FileHandler(
            f"Slurm/{dataConfig.dataset_name}/{modelConfig.model_name}/{str(time.time())}.log"
        )
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts **********")
        logger.info(f"Configuration: {vars(modelConfig)} | Data: {vars(dataConfig)}")

        # ---------------------
        # MODEL INITIALIZATION
        # ---------------------
        # Choose different architectures depending on modelConfig.model_name
        if modelConfig.model_name == 'TGAT':
            dynamic_backbone = TGAT(
                num_nodes=node_raw_features.shape[0],
                node_dim=node_raw_features.shape[1],
                edge_dim=edge_raw_features.shape[1],
                time_feat_dim=modelConfig.time_feat_dim,
                num_layers=modelConfig.num_layers,
                num_heads=modelConfig.num_heads,
                dropout=modelConfig.dropout,
                device=modelConfig.device
            )

        elif modelConfig.model_name in ['JODIE', 'DyRep', 'TGN']:
            # Compute time-shift statistics (needed for Memory Models)
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(
                    train_data.src_node_ids,
                    train_data.dst_node_ids,
                    train_data.node_interact_times
                )
            dynamic_backbone = MemoryModel(
                num_nodes=node_raw_features.shape[0],
                node_dim=node_raw_features.shape[1],
                edge_dim=edge_raw_features.shape[1],
                time_feat_dim=modelConfig.time_feat_dim,
                model_name=modelConfig.model_name,
                num_layers=modelConfig.num_layers,
                num_heads=modelConfig.num_heads,
                dropout=modelConfig.dropout,
                src_node_mean_time_shift=src_node_mean_time_shift,
                src_node_std_time_shift=src_node_std_time_shift,
                dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst,
                dst_node_std_time_shift=dst_node_std_time_shift,
                device=modelConfig.device
            )

        elif modelConfig.model_name == 'CAWN':
            dynamic_backbone = CAWN(
                num_nodes=node_raw_features.shape[0],
                node_dim=node_raw_features.shape[1],
                edge_dim=edge_raw_features.shape[1],
                time_feat_dim=modelConfig.time_feat_dim,
                position_feat_dim=modelConfig.position_feat_dim,
                walk_length=modelConfig.walk_length,
                num_walk_heads=modelConfig.num_walk_heads,
                dropout=modelConfig.dropout,
                device=modelConfig.device
            )

        elif modelConfig.model_name == 'TCL':
            dynamic_backbone = TCL(
                num_nodes=node_raw_features.shape[0],
                node_dim=node_raw_features.shape[1],
                edge_dim=edge_raw_features.shape[1],
                time_feat_dim=modelConfig.time_feat_dim,
                num_layers=modelConfig.num_layers,
                num_heads=modelConfig.num_heads,
                num_depths=modelConfig.num_neighbors + 1,
                dropout=modelConfig.dropout,
                device=modelConfig.device
            )

        elif modelConfig.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(
                num_nodes=node_raw_features.shape[0],
                node_dim=node_raw_features.shape[1],
                edge_dim=edge_raw_features.shape[1],
                time_feat_dim=modelConfig.time_feat_dim,
                num_tokens=modelConfig.num_neighbors,
                time_gap=modelConfig.time_gap,
                num_layers=modelConfig.num_layers,
                dropout=modelConfig.dropout,
                device=modelConfig.device
            )

        elif modelConfig.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(
                num_nodes=node_raw_features.shape[0],
                node_dim=node_raw_features.shape[1],
                edge_dim=edge_raw_features.shape[1],
                time_feat_dim=modelConfig.time_feat_dim,
                channel_embedding_dim=modelConfig.channel_embedding_dim,
                patch_size=modelConfig.patch_size,
                num_layers=modelConfig.num_layers,
                num_heads=modelConfig.num_heads,
                dropout=modelConfig.dropout,
                max_input_sequence_length=modelConfig.max_input_sequence_length,
                device=modelConfig.device
            )

        else:
            raise ValueError(f"Unknown model: {modelConfig.model_name}")

        # ---------------------
        # DATA LOADERS (indices only, features gathered later)
        # ---------------------
        train_idx_data_loader = get_idx_data_loader(
            np.where(~np.isnan(train_data.labels))[0],
            batch_size=trainConfig.batch_size,
            shuffle=not dynamic_backbone.is_statefull
        )
        val_idx_data_loader = get_idx_data_loader(
            np.where(~np.isnan(val_data.labels))[0],
            batch_size=trainConfig.batch_size,
            shuffle=not dynamic_backbone.is_statefull
        )
        test_idx_data_loader = get_idx_data_loader(
            np.where(~np.isnan(test_data.labels))[0],
            batch_size=trainConfig.batch_size,
            shuffle=not dynamic_backbone.is_statefull
        )

        # ---------------------
        # CREATE FINAL MODEL (TGNN wrapper + regression head)
        # ---------------------
        regressor = NeuralNetworkSrcDst(
            input_dim=node_raw_features.shape[1],
            num_layers=modelConfig.num_reg_layers,
            hidden_dim=modelConfig.hidden_reg_layers_dim
        )
        model = TGNN(dynamic_backbone, regressor)

        logger.info(f"Model Summary: {model}")
        logger.info(f"Total Params: {get_parameter_sizes(model)} elements | "
                    f"{get_parameter_sizes(model) * 4 / 1024 / 1024:.4f} MB")

        optimizer = create_optimizer(
            model=model,
            optimizer_name=trainConfig.optimizer,
            learning_rate=trainConfig.learning_rate,
            weight_decay=trainConfig.weight_decay
        )
        model.to(modelConfig.device)

        # Ensure memory-based model raw messages are moved to device
        if isinstance(model.backbone, MemoryModel):
            for node_id, node_raw_messages in model.backbone.memory_bank.node_raw_messages.items():
                model.backbone.memory_bank.node_raw_messages[node_id] = [
                    (msg[0].to(modelConfig.device), msg[1]) for msg in node_raw_messages
                ]

        # ---------------------
        # Checkpoint / EarlyStopping
        # ---------------------
        save_model_folder = f"Saved_models/{dataConfig.dataset_name}/{modelConfig.model_name}/seed{run}"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(
            patience=trainConfig.patience,
            save_model_folder=save_model_folder,
            save_model_name=model_name,
            logger=logger,
            model_name=modelConfig.model_name
        )
        
        loss_func = nn.MSELoss()

        # ---------------------
        # MAIN EPOCH TRAINING LOOP
        # ---------------------
        for epoch in range(trainConfig.num_epochs):
            model.train()

            # For memory-based models, reset the memory bank at the start of each epoch
            if isinstance(model.backbone, MemoryModel):
                model.backbone.memory_bank.__init_memory_bank__()

            train_y_trues, train_y_predicts = [], []

            # tqdm progress bar for training
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)

            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()

                # Gather batch info
                batch_src_node_ids = train_data.src_node_ids[train_data_indices]
                batch_dst_node_ids = train_data.dst_node_ids[train_data_indices]
                batch_node_interact_times = train_data.node_interact_times[train_data_indices]
                batch_edge_ids = train_data.edge_ids[train_data_indices]
                batch_labels = train_data.labels[train_data_indices]

                labels = torch.from_numpy(batch_labels).float()
                not_nan_mask = ~torch.isnan(labels)  # avoid NaN labels
                
                loss = torch.tensor(np.nan)  # placeholder loss

                if not_nan_mask.any():
                    # Sample temporal subgraphs for src and dst nodes
                    subgraphs_src = full_neighbor_sampler.get_multi_hop_neighbors(
                        modelConfig.num_layers,
                        batch_src_node_ids[not_nan_mask],
                        batch_node_interact_times[not_nan_mask],
                        num_neighbors=modelConfig.num_neighbors
                    )
                    subgraphs_dst = full_neighbor_sampler.get_multi_hop_neighbors(
                        modelConfig.num_layers,
                        batch_dst_node_ids[not_nan_mask],
                        batch_node_interact_times[not_nan_mask],
                        num_neighbors=modelConfig.num_neighbors
                    )

                    # Gather edge features for subgraph edges
                    edge_feat_src = full_neighbor_sampler.get_edge_features_for_multi_hop(subgraphs_src[1])
                    edge_feat_dst = full_neighbor_sampler.get_edge_features_for_multi_hop(subgraphs_dst[1])

                    # BatchSubgraphs container for model input
                    subgraphs_src = BatchSubgraphs(*subgraphs_src, edge_feat_src)
                    subgraphs_src.to(modelConfig.device)
                    subgraphs_dst = BatchSubgraphs(*subgraphs_dst, edge_feat_dst)
                    subgraphs_dst.to(modelConfig.device)

                    # Forward pass
                    predicts = model(
                        src_node_ids=batch_src_node_ids[not_nan_mask],
                        dst_node_ids=batch_dst_node_ids[not_nan_mask],
                        node_interact_times=batch_node_interact_times[not_nan_mask],
                        src_subgraphs=subgraphs_src,
                        dst_subgraphs=subgraphs_dst,
                        num_neighbors=modelConfig.num_neighbors,
                        edge_ids=batch_edge_ids[not_nan_mask],
                        edges_are_positive=False
                    ).squeeze(dim=-1)

                    # Compute loss
                    labels = torch.from_numpy(batch_labels).float().to(predicts.device)[not_nan_mask]
                    loss = loss_func(input=predicts, target=labels)

                    train_y_trues.append(labels)
                    train_y_predicts.append(predicts)

                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Update tqdm every 50 batches
                if batch_idx % 50 == 0:
                    train_idx_data_loader_tqdm.set_description(
                        f"Epoch: {epoch + 1}, Batch: {batch_idx + 1}, Loss: {loss.item() if not torch.isnan(loss) else np.nan}"
                    )

            # ---------------------
            # EVALUATE ON VALIDATION
            # ---------------------
            train_metrics = get_link_regression_metrics(
                predicts=torch.cat(train_y_predicts),
                true_value=torch.cat(train_y_trues)
            )
            val_metrics = evaluate_model_link_regression(
                model=model,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=val_idx_data_loader,
                evaluate_data=val_data,
                loss_func=loss_func,
                num_neighbors=modelConfig.num_neighbors,
                num_layers=modelConfig.num_layers,
                time_gap=modelConfig.time_gap,
                device=modelConfig.device
            )

            logger.info(f"Epoch: {epoch + 1}, LR: {optimizer.param_groups[0]['lr']}")
            for m in train_metrics: logger.info(f"train {m}: {train_metrics[m]:.4f}")
            for m in val_metrics: logger.info(f"val {m}: {val_metrics[m]:.4f}")

            # ---------------------
            # OPTIONAL TEST EVALUATION
            # ---------------------
            if (epoch + 1) % trainConfig.test_interval_epochs == 0:
                # Backup memory for memory-based models before test
                if isinstance(model.backbone, MemoryModel):
                    val_backup_memory_bank = model.backbone.memory_bank.backup_memory_bank()

                test_metrics = evaluate_model_link_regression(
                    model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=test_idx_data_loader,
                    evaluate_data=test_data,
                    loss_func=loss_func,
                    num_neighbors=modelConfig.num_neighbors,
                    num_layers=modelConfig.num_layers,
                    time_gap=modelConfig.time_gap,
                    device=modelConfig.device
                )

                # Restore validation memory
                if isinstance(model.backbone, MemoryModel):
                    model.backbone.memory_bank.reload_memory_bank(val_backup_memory_bank)

                for m in test_metrics: logger.info(f"test {m}: {test_metrics[m]:.4f}")

            # ---------------------
            # EARLY STOPPING CHECK
            # ---------------------
            val_metric_indicator = [(list(val_metrics.keys())[0], list(val_metrics.values())[0], False)]
            if early_stopping.step(val_metric_indicator, model, val_metrics, save_model_folder, model_name):
                break

        # ---------------------
        # POST-TRAIN: LOAD BEST MODEL & FINAL EVALUATION
        # ---------------------
        early_stopping.load_checkpoint(model)

        logger.info(f"Final evaluation for dataset {dataConfig.dataset_name}...")

        train_metric_dict, val_metric_dict, test_metric_dict = {}, {}, {}

        if not isinstance(model.backbone, MemoryModel):
            # Train evaluation
            train_metrics = evaluate_model_link_regression(
                model=model,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=train_idx_data_loader,
                evaluate_data=train_data,
                loss_func=loss_func,
                num_neighbors=modelConfig.num_neighbors,
                num_layers=modelConfig.num_layers,
                time_gap=modelConfig.time_gap,
                device=modelConfig.device
            )
            train_metric_dict.update(train_metrics)

            # Validation evaluation
            val_metrics = evaluate_model_link_regression(
                model=model,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=val_idx_data_loader,
                evaluate_data=val_data,
                loss_func=loss_func,
                num_neighbors=modelConfig.num_neighbors,
                num_layers=modelConfig.num_layers,
                time_gap=modelConfig.time_gap,
                device=modelConfig.device
            )
            val_metric_dict.update(val_metrics)

        # Test evaluation
        test_metrics = evaluate_model_link_regression(
            model=model,
            neighbor_sampler=full_neighbor_sampler,
            evaluate_idx_data_loader=test_idx_data_loader,
            evaluate_data=test_data,
            loss_func=loss_func,
            num_neighbors=modelConfig.num_neighbors,
            num_layers=modelConfig.num_layers,
            time_gap=modelConfig.time_gap,
            device=modelConfig.device
        )
        test_metric_dict.update(test_metrics)

        # Timing
        single_run_time = time.time() - run_start_time
        logger.info(f"Run {run + 1} took {single_run_time:.2f} seconds.")

        # Save run results
        if not isinstance(model.backbone, MemoryModel):
            val_metric_all_runs.append(val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)

        if run < trainConfig.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # Save results to JSON
        result_json = {
            "train metrics": {m: f"{train_metric_dict[m]:.4f}" for m in train_metric_dict},
            "validate metrics": {m: f"{val_metric_dict[m]:.4f}" for m in val_metric_dict},
            "test metrics": {m: f"{test_metric_dict[m]:.4f}" for m in test_metric_dict}
        } if not isinstance(model.backbone, MemoryModel) else {
            "test metrics": {m: f"{test_metric_dict[m]:.4f}" for m in test_metric_dict}
        }
        with open(os.path.join(save_model_folder, f"{model_name}.json"), 'w') as f:
            f.write(json.dumps(result_json, indent=4))

    # ---------------------
    # AGGREGATE RESULTS OVER RUNS
    # ---------------------
    logger.info(f"Metrics over {trainConfig.num_runs} runs:")
    if modelConfig.model_name not in ['JODIE', 'DyRep', 'TGN']:
        for m in val_metric_all_runs[0]:
            scores = [run_metrics[m] for run_metrics in val_metric_all_runs]
            logger.info(f"validate {m}: {scores}")
            logger.info(f"avg validate {m}: {np.mean(scores):.4f} ± {np.std(scores, ddof=1):.4f}")

    for m in test_metric_all_runs[0]:
        scores = [run_metrics[m] for run_metrics in test_metric_all_runs]
        logger.info(f"test {m}: {scores}")
        logger.info(f"avg test {m}: {np.mean(scores):.4f} ± {np.std(scores, ddof=1):.4f}")

    return val_metric_all_runs, test_metric_all_runs

