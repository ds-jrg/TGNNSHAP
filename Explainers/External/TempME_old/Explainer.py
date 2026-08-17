from typing import Tuple

from Explainers.utils import Explainer, ExplanationResult

from DyGLib.utils.DataLoader import Data
from DyGLib.utils.utils import NeighborSampler, NegativeEdgeSampler, BatchSubgraphs, concat_subgraphs
from DyGLib.models.modules import TGNN

from Config.config import CONFIG

from .utils import load_subgraph_margin, get_item, get_item_edge, WalkFinder
from .processed.data_preprocess import pre_processing, calculate_edge
from .models import TempME

import numpy as np
import h5py
import math
import torch
from tqdm import tqdm
from abc import abstractmethod
import os

CONFIG = CONFIG()

class TempMEExplainer(Explainer):
    def __init__(self, model:TGNN, neighbor_finder: NeighborSampler, data: Data):
        super().__init__(model, neighbor_finder, data)
        self.explainer = TempME(model, neighbor_finder=neighbor_finder,
                                data=CONFIG.data.dataset_name, out_dim=CONFIG.tempME.out_dim, hid_dim=CONFIG.tempME.hid_dim,
                                temp=CONFIG.tempME.temp, if_cat_feature=True,
                                dropout_p=CONFIG.tempME.drop_out, device=CONFIG.tempME.device)

        if(np.isnan(self.data.labels).any()):
            mask = ~np.isnan(self.data.labels)
            self.edges = self.data.edge_ids[mask]
        else:
            self.edges = self.data.edge_ids


    def preprocess(self,
                   full_finder: WalkFinder,
                   full_sampler: NegativeEdgeSampler, train=True):
        if(np.isnan(self.data.labels).any()):
            mask = ~np.isnan(self.data.labels)
        else:
            mask = np.ones_like(self.data.edge_ids, dtype="bool")

        mode = "train" if train else "full"

        pre_processing(full_finder, self.neighbor_finder, full_sampler, 
                       self.data.src_node_ids[mask], self.data.dst_node_ids[mask], 
                       self.data.node_interact_times[mask], val_e_idx_l=None, 
                       MODE=mode, data=CONFIG.data.dataset_name, num_neig=CONFIG.model.num_neighbors)
        
        data_path = f'{CONFIG.data.folder}/TempME/{CONFIG.data.dataset_name}_{mode}.h5'
        file = h5py.File(data_path,'r')
        file.keys()
        walks_src = file["walks_src"][:] # type: ignore
        walks_tgt = file["walks_tgt"][:] # type: ignore
        walks_bgd = file["walks_bgd"][:] # type: ignore
        file.close()
        edge_load = calculate_edge(walks_src, walks_tgt, walks_bgd)
        save_path = f"{CONFIG.data.folder}/TempME/{CONFIG.data.dataset_name}_{mode}_edge.npy"
        np.save(save_path, edge_load)

    def initialize(self, train=False):
        mode = "train" if train else "full"

        pre_load = h5py.File(f'{CONFIG.data.folder}/TempME/{CONFIG.data.dataset_name}_{mode}.h5','r')

        self.pack = load_subgraph_margin(CONFIG.model.num_neighbors, pre_load)

        pre_load = None

        self.edge = np.load(f"{CONFIG.data.folder}/TempME/{CONFIG.data.dataset_name}_{mode}_edge.npy")

        self.explainer = self.explainer.to(CONFIG.tempME.device)
        os.makedirs(f"Saved_models/{CONFIG.data.dataset_name}/TempMe", exist_ok=True)
        if train:
            self.train()
        else:
            self.explainer.load_state_dict(torch.load(f"Saved_models/{CONFIG.data.dataset_name}/TempMe/Explainer.pt", weights_only=True))

        
    def train(self):
        optimizer = torch.optim.Adam(self.explainer.parameters(),
                                        lr=CONFIG.tempME.lr,
                                        betas=(0.9, 0.999), eps=1e-8,
                                        weight_decay=CONFIG.tempME.weight_decay)
        
        if CONFIG.model.task == "regression":
            criterion = torch.nn.MSELoss()
        else: 
            criterion = torch.nn.BCEWithLogitsLoss()

        if(np.isnan(self.data.labels).any()):
            mask = ~np.isnan(self.data.labels)
        else:
            mask = np.ones_like(self.data.edge_ids, dtype="bool")


        src_l, dst_l, ts_l, label_l, e_idx_l = self.data.src_node_ids[mask], self.data.dst_node_ids[mask], self.data.node_interact_times[mask], self.data.labels[mask], self.data.edge_ids[mask]
    
        num_instance = len(src_l)
        num_batch = math.ceil(num_instance / CONFIG.tempME.bs)
        best_acc = 0
        print('num of training instances: {}'.format(num_instance))
        print('num of batches per epoch: {}'.format(num_batch))
        idx_list = np.arange(num_instance)
        np.random.shuffle(idx_list)

        for epoch in range(CONFIG.tempME.n_epoch):
            train_loss = []
            train_pred_loss = []
            train_kl_loss = []
            np.random.shuffle(idx_list)
            self.explainer.train()
            for k in tqdm(range(num_batch)):
                s_idx = k * CONFIG.tempME.bs
                e_idx = min(num_instance, s_idx + CONFIG.tempME.bs)
                if s_idx == e_idx:
                    continue
                batch_idx = idx_list[s_idx:e_idx]
                batch_size = len(batch_idx)
                src_l_cut, dst_l_cut = src_l[batch_idx], dst_l[batch_idx]
                ts_l_cut = ts_l[batch_idx]
                e_l_cut = e_idx_l[batch_idx]
                subgraph_src, subgraph_tgt, subgraph_bgd, walks_src, walks_tgt, walks_bgd, dst_l_fake = get_item(self.pack,
                                                                                                                    batch_idx)
                src_edge, tgt_edge, bgd_edge = get_item_edge(self.edge, batch_idx)
                with torch.no_grad():
                    edge_feat_src = self.neighbor_finder.get_edge_features_for_multi_hop(subgraph_src[1])
                    subgraphs_src = BatchSubgraphs(*subgraph_src, edge_feat_src)
                    subgraphs_src.chop_layers(CONFIG.model.num_layers)
                    subgraphs_src.to(CONFIG.model.device)
                    
                    edge_feat_dst = self.neighbor_finder.get_edge_features_for_multi_hop(subgraph_tgt[1])
                    subgraphs_dst = BatchSubgraphs(*subgraph_tgt, edge_feat_dst)
                    subgraphs_dst.chop_layers(CONFIG.model.num_layers)
                    subgraphs_dst.to(CONFIG.model.device)

                    y_ori = self.model(src_node_ids=src_l_cut,
                                        dst_node_ids=dst_l_cut,
                                        node_interact_times=ts_l_cut,
                                        src_subgraphs=subgraphs_src, dst_subgraphs=subgraphs_dst,
                                        num_neighbors=CONFIG.model.num_neighbors,
                                        time_gap=CONFIG.model.time_gap,
                                        edge_ids=e_l_cut,
                                        edges_are_positive=False).squeeze(dim=-1)

                optimizer.zero_grad()
                graphlet_imp_src = self.explainer(walks_src, ts_l_cut, src_edge)
                graphlet_imp_tgt = self.explainer(walks_tgt, ts_l_cut, tgt_edge)
                graphlet_imp_bgd = self.explainer(walks_bgd, ts_l_cut, bgd_edge)
                explanation = self.explainer.retrieve_explanation(subgraph_src, graphlet_imp_src, walks_src,
                                                            subgraph_tgt, graphlet_imp_tgt, walks_tgt,
                                                            subgraph_bgd, graphlet_imp_bgd, walks_bgd,
                                                            training=CONFIG.tempME.if_bern)
                
                subgraphs_src.set_event_attention([explanation[0][0:batch_size], explanation[1][0:batch_size]])
                subgraphs_dst.set_event_attention([explanation[0][batch_size:2 * batch_size], explanation[1][batch_size:2 * batch_size]])

                with torch.no_grad():
                    pred = self.model(src_node_ids=src_l_cut,
                                        dst_node_ids=dst_l_cut,
                                        node_interact_times=ts_l_cut,
                                        src_subgraphs = subgraphs_src, 
                                        dst_subgraphs = subgraphs_dst, 
                                        num_neighbors=CONFIG.model.num_neighbors, time_gap=CONFIG.model.time_gap, edges_are_positive = False).squeeze(dim=-1)
                    
                pred_loss = criterion(pred, y_ori)
                kl_loss = self.explainer.kl_loss(graphlet_imp_src, walks_src, target=CONFIG.tempME.prior_p) + \
                            self.explainer.kl_loss(graphlet_imp_tgt, walks_tgt, target=CONFIG.tempME.prior_p) + \
                            self.explainer.kl_loss(graphlet_imp_bgd, walks_bgd, target=CONFIG.tempME.prior_p)
                loss = pred_loss + CONFIG.tempME.beta * kl_loss
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    train_loss.append(loss.item())
                    train_pred_loss.append(pred_loss.item())
                    train_kl_loss.append(kl_loss.item())

            loss_epoch = np.mean(train_loss)
            pred_loss_epoch = np.mean(train_pred_loss)
            kl_epoch = np.mean(train_kl_loss)
            print((f'Training Epoch: {epoch} | '
                    f'Training loss: {loss_epoch} | '
                    f'Pred loss: {pred_loss_epoch} | '
                    f'KL loss: {kl_epoch} | '))
        
        torch.save(self.explainer.state_dict(), f"Saved_models/{CONFIG.data.dataset_name}/TempMe/Explainer.pt")

    def explain_instance(self, src, dst, timestamp, silent = False):
        mask = (self.data.src_node_ids == src) & (self.data.dst_node_ids == dst) & (self.data.node_interact_times == timestamp)


        edge_id = self.data.edge_ids[mask][0]
        edge_id = np.where(self.edges == edge_id)[0]

        subgraph_src, subgraph_tgt, subgraph_bgd, walks_src, walks_tgt, walks_bgd, dst_l_fake = get_item(self.pack, edge_id)
        src_edge, tgt_edge, bgd_edge = get_item_edge(self.edge, edge_id)
        
        timestamp = np.array([timestamp])
        graphlet_imp_src = self.explainer(walks_src, timestamp, src_edge)
        graphlet_imp_tgt = self.explainer(walks_tgt, timestamp, tgt_edge)
        graphlet_imp_bgd = self.explainer(walks_bgd, timestamp, bgd_edge)
        explanation = self.explainer.retrieve_explanation(subgraph_src, graphlet_imp_src, walks_src,
                                                    subgraph_tgt, graphlet_imp_tgt, walks_tgt,
                                                    subgraph_bgd, graphlet_imp_bgd, walks_bgd,
                                                    training=False)
        edge_feat_src = self.neighbor_finder.get_edge_features_for_multi_hop(subgraph_src[1])
        edge_feat_dst = self.neighbor_finder.get_edge_features_for_multi_hop(subgraph_tgt[1])

        for i, _ in enumerate(explanation):
            explanation[i] = explanation[i].detach().cpu()

        src_subgraph = BatchSubgraphs(subgraph_src[0], subgraph_src[1], subgraph_src[2], 
                                                edge_feat_src, event_attention=[explanation[0][0:1], explanation[1][0:1]])
        dst_subgraph = BatchSubgraphs(subgraph_tgt[0], subgraph_tgt[1], subgraph_tgt[2], 
                                         edge_feat_dst, event_attention=[explanation[0][1:2], explanation[1][1:2]])
        
        src_subgraph.chop_layers(CONFIG.model.num_layers)
        dst_subgraph.chop_layers(CONFIG.model.num_layers)

        full_subgraph = concat_subgraphs([src_subgraph, dst_subgraph])
        return full_subgraph, src_subgraph, dst_subgraph
    
    def build_coalitions(self, explanation: Tuple[BatchSubgraphs,BatchSubgraphs,BatchSubgraphs]):
        full_subgraph, src_subgraph, dst_subgraph = explanation
        edges = np.concat(full_subgraph.events, axis=1).flatten()
        attentions = torch.concat(full_subgraph.event_attention, dim=1).flatten()
        
        
        shuffle_mask = torch.randperm(len(edges))
        edges = edges[shuffle_mask]
        attentions = attentions[shuffle_mask]

        mask = edges != 0
        edges = edges[np.where(mask)]
        attentions = attentions[mask]

        sorting = (-attentions).argsort()

        if(len(edges) > 1):
            edges = edges[sorting.cpu()]
        
        coalitions = np.zeros((edges.shape[0], edges.shape[0]))
        unique_edges = np.zeros((0,))
        for i in range(edges.shape[0]):
            unique_edges = np.unique(edges[:i+1])
            coalitions[i, :unique_edges.shape[0]] = unique_edges

        result = coalitions[:, :unique_edges.shape[0]]

        for i in range(src_subgraph.get_num_layers()):
            src_subgraph.event_attention[i][:,:] = 1.0
        
        for i in range(dst_subgraph.get_num_layers()):
            dst_subgraph.event_attention[i][:,:] = 1.0

        src_subgraph.to("cpu")
        dst_subgraph.to("cpu")
        
        return result, src_subgraph, dst_subgraph