import math
import logging
import time
import h5py
import numpy as np
import random
import sys
from tqdm import tqdm
import os.path as osp
import pickle
import torch
import pandas as pd
import numpy as np
from utils import NeighborFinder

degree_dict = {"wikipedia":20, "reddit":20 ,"uci":30 ,"mooc":60, "enron": 30, "canparl": 30, "uslegis": 30}

data = "wikipedia"
NUM_NEIGHBORS = degree_dict[data]


def load_data(mode, data):
    g_df = pd.read_csv(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'processed/ml_{}.csv'.format(data)))
    val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))
    src_l = g_df.u.values
    dst_l = g_df.i.values
    e_idx_l = g_df.idx.values
    label_l = g_df.label.values
    ts_l = g_df.ts.values
    max_src_index = src_l.max()
    max_idx = max(src_l.max(), dst_l.max())
    random.seed(2023)
    total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
    num_total_unique_nodes = len(total_node_set)
    mask_node_set = set(random.sample(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])),
                                      int(0.1 * num_total_unique_nodes)))
    mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
    mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
    none_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)
    valid_train_flag = (ts_l <= val_time) * (none_node_flag > 0)
    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    train_ts_l = ts_l[valid_train_flag]
    train_e_idx_l = e_idx_l[valid_train_flag]
    train_label_l = label_l[valid_train_flag]
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
    valid_test_flag = ts_l > test_time
    val_src_l = src_l[valid_val_flag]
    val_dst_l = dst_l[valid_val_flag]
    test_src_l = src_l[valid_test_flag]
    test_dst_l = dst_l[valid_test_flag]
    test_ts_l = ts_l[valid_test_flag]
    test_e_idx_l = e_idx_l[valid_test_flag]
    test_label_l = label_l[valid_test_flag]
    adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
        adj_list[src].append((dst, eidx, ts))
        adj_list[dst].append((src, eidx, ts))
    train_ngh_finder = NeighborFinder(adj_list)
    # full graph with all the data for the test and validation purpose
    full_adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
        full_adj_list[src].append((dst, eidx, ts))
        full_adj_list[dst].append((src, eidx, ts))
    full_ngh_finder = NeighborFinder(full_adj_list)
    train_rand_sampler = RandEdgeSampler((train_src_l,), (train_dst_l,))
    # val_rand_sampler = RandEdgeSampler((train_src_l, val_src_l), (train_dst_l, val_dst_l))
    test_rand_sampler = RandEdgeSampler((train_src_l, val_src_l, test_src_l), (train_dst_l, val_dst_l, test_dst_l))
    if mode == "test":
        return test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l, full_ngh_finder
    else:
        return train_rand_sampler, train_src_l, train_dst_l, train_ts_l, train_label_l, train_e_idx_l, train_ngh_finder


def statistic(out_anony):
    batch = out_anony.shape[0]
    sat = {}
    strint_rep = {}
    for item in ["1,2,1","1,2,2","1,2,3","1,2,0","1,3,1","1,3,3","1,3,2","1,3,0","1,1,3","1,1,2","1,1,1", "1,1,0"]:
        sat[item] = 0
        array_rep = np.array(list(eval(item)))
        string = np.array2string(array_rep)
        strint_rep[string] = item
    for i in range(batch):
        samples = out_anony[i]  #[N**2, 3]
        for t in range(samples.shape[0]):
            anony_string = np.array2string(samples[t])
            sat[strint_rep[anony_string]] += 1
    for key, value in sat.items():
        sat[key] = value/batch
    return sat



def pre_processing(full_ngh_finder, sampler, src, dst, ts, val_e_idx_l, MODE="test", data="reddit"):
    load_dict = {}
    save_dict = {}
    for item in ["subgraph_src_0", "subgraph_src_1", "subgraph_tgt_0", "subgraph_tgt_1",  "subgraph_bgd_0", "subgraph_bgd_1", "walks_src", "walks_tgt", "walks_bgd", "dst_fake"]:
        load_dict[item] = []
    num_test_instance = len(src)
    print("start extracting subgraph")
    for k in tqdm(range(num_test_instance-1)):
        src_l_cut = src[k:k+1]
        dst_l_cut = dst[k:k+1]
        ts_l_cut = ts[k:k+1]
        e_l_cut = val_e_idx_l[k:k+1] if (val_e_idx_l is not None) else None
        size = len(src_l_cut)
        src_l_fake, dst_l_fake = sampler.sample(size)
        load_dict["dst_fake"].append(dst_l_fake)
        subgraph_src = ngh_finder.find_k_hop(src_l_cut, ts_l_cut, e_idx_l=e_l_cut)  #first: (batch, num_neighbors), second: [batch, num_neighbors * num_neighbors]
        node_records, eidx_records, t_records = subgraph_src
        load_dict["subgraph_src_0"].append(np.concatenate([node_records[0], eidx_records[0], t_records[0]], axis=-1))  #append([1, num_neighbors * 3]
        load_dict["subgraph_src_1"].append(np.concatenate([node_records[1], eidx_records[1], t_records[1]], axis=-1))    #append([1, num_neighbors**2 * 3]
        subgraph_tgt = ngh_finder.find_k_hop(dst_l_cut, ts_l_cut, e_idx_l=e_l_cut)
        node_records, eidx_records, t_records = subgraph_tgt
        load_dict["subgraph_tgt_0"].append(np.concatenate([node_records[0], eidx_records[0], t_records[0]], axis=-1))  #append([1, num_neighbors * 3]
        load_dict["subgraph_tgt_1"].append(np.concatenate([node_records[1], eidx_records[1], t_records[1]], axis=-1))    #append([1, num_neighbors**2 * 3]
        subgraph_bgd = ngh_finder.find_k_hop(dst_l_fake, ts_l_cut, e_idx_l=None)
        node_records, eidx_records, t_records = subgraph_bgd
        load_dict["subgraph_bgd_0"].append(np.concatenate([node_records[0], eidx_records[0], t_records[0]], axis=-1))  #append([1, num_neighbors * 3]
        load_dict["subgraph_bgd_1"].append(np.concatenate([node_records[1], eidx_records[1], t_records[1]], axis=-1))    #append([1, num_neighbors**2 * 3]
        walks_src = ngh_finder.find_k_walks(NUM_NEIGHBORS, src_l_cut, num_neighbors=3, subgraph_src=subgraph_src)
        walks_tgt = ngh_finder.find_k_walks(NUM_NEIGHBORS, dst_l_cut, num_neighbors=3, subgraph_src=subgraph_tgt)
        walks_bgd = ngh_finder.find_k_walks(NUM_NEIGHBORS, dst_l_fake, num_neighbors=3, subgraph_src=subgraph_bgd)
        node_records, eidx_records, t_records, out_anony = walks_src
        load_dict["walks_src"].append(np.concatenate([node_records, eidx_records, t_records, out_anony], axis=-1))  #append([1, num_walks, 6+3+3+3])
        node_records, eidx_records, t_records, out_anony = walks_tgt
        load_dict["walks_tgt"].append(np.concatenate([node_records, eidx_records, t_records, out_anony], axis=-1))
        node_records, eidx_records, t_records, out_anony = walks_bgd
        load_dict["walks_bgd"].append(np.concatenate([node_records, eidx_records, t_records, out_anony], axis=-1))
    for item in ["subgraph_src_0", "subgraph_src_1", "subgraph_tgt_0", "subgraph_tgt_1", "subgraph_bgd_0",
                 "subgraph_bgd_1", "walks_src", "walks_tgt", "walks_bgd", "dst_fake"]:
        save_dict[item] = np.concatenate(load_dict[item], axis=0)

    hf = h5py.File(f"{data}_{MODE}.h5", "w")
    for item in ["subgraph_src_0", "subgraph_src_1", "subgraph_tgt_0", "subgraph_tgt_1", "subgraph_bgd_0",
                 "subgraph_bgd_1", "walks_src", "walks_tgt", "walks_bgd","dst_fake"]:
        hf.create_dataset(item, data=save_dict[item])
    hf.close()
    print("done")
    return


def marginal(walks_src, walks_tgt, walks_bgd):
    '''
    :param walks: [data_size, num_walks, 6+3+3+3]
    :return: [data_size, num_walks, 6+3+3+1]
    '''
    node_records_src, eidx_records_src, t_records_src, out_anony_src = walks_src[:, :, :6], walks_src[:, :, 6:9], walks_src[:, :,9:12], walks_src[:, :,12:15]
    node_records_tgt, eidx_records_tgt, t_records_tgt, out_anony_tgt = walks_tgt[:, :, :6], walks_tgt[:, :, 6:9], walks_tgt[:, :, 9:12], walks_tgt[:, :, 12:15]
    node_records_bgd, eidx_records_bgd, t_records_bgd, out_anony_bgd = walks_bgd[:, :, :6], walks_bgd[:, :, 6:9], walks_bgd[:, :, 9:12], walks_bgd[:, :, 12:15]
    out_anony_src = out_anony_src.astype(int)
    out_anony_tgt = out_anony_tgt.astype(int)
    out_anony_bgd = out_anony_bgd.astype(int)
    num_data = out_anony_src.shape[0]
    num_walk_per_data = out_anony_src.shape[1]
    marginal_repr_src = np.empty((num_data, num_walk_per_data, 1))
    marginal_repr_tgt = np.empty((num_data, num_walk_per_data, 1))
    marginal_repr_bgd = np.empty((num_data, num_walk_per_data, 1))
    cate_feat_src = np.empty((num_data, num_walk_per_data, 1))
    cate_feat_tgt = np.empty((num_data, num_walk_per_data, 1))
    cate_feat_bgd = np.empty((num_data, num_walk_per_data, 1))
    sat = {}
    strint_rep = {}
    strint_id = {}
    t = 0
    for item in ["1,2,1", "1,2,2", "1,2,3", "1,2,0", "1,3,1", "1,3,3", "1,3,2", "1,3,0", "1,1,3", "1,1,2", "1,1,1",
                 "1,1,0"]:
        sat[item] = 0
        array_rep = np.array(list(eval(item)))
        string = np.array2string(array_rep)
        strint_rep[string] = item
        strint_id[string] = t
        t = t + 1

    for i in range(num_data):
        samples_src = out_anony_src[i]  # [N**2, 3]
        samples_tgt = out_anony_tgt[i]  # [N**2, 3]
        samples_bgd = out_anony_bgd[i]  # [N**2, 3]
        for t in range(samples_src.shape[0]):
            anony_string = np.array2string(samples_src[t])
            sat[strint_rep[anony_string]] += 1
            anony_string = np.array2string(samples_tgt[t])
            sat[strint_rep[anony_string]] += 1
            anony_string = np.array2string(samples_bgd[t])
            sat[strint_rep[anony_string]] += 1
    for key, value in sat.items():
        sat[key] = value / (num_data * num_walk_per_data * 3)
        print(key, sat[key])

    for i in tqdm(range(num_data)):
        samples_src = out_anony_src[i]  # [N**2, 3]
        samples_tgt = out_anony_tgt[i]  # [N**2, 3]
        samples_bgd = out_anony_bgd[i]  # [N**2, 3]
        for t in range(samples_src.shape[0]):
            anony_string = np.array2string(samples_src[t])
            marginal_repr_src[i, t] = sat[strint_rep[anony_string]]
            cate_feat_src[i, t] = strint_id[anony_string]
            anony_string = np.array2string(samples_tgt[t])
            marginal_repr_tgt[i, t] = sat[strint_rep[anony_string]]
            cate_feat_tgt[i, t] = strint_id[anony_string]
            anony_string = np.array2string(samples_bgd[t])
            marginal_repr_bgd[i, t] = sat[strint_rep[anony_string]]
            cate_feat_bgd[i, t] = strint_id[anony_string]


    walks_src_new = np.concatenate([node_records_src, eidx_records_src, t_records_src, cate_feat_src, marginal_repr_src], axis=-1)
    walks_tgt_new = np.concatenate([node_records_tgt, eidx_records_tgt, t_records_tgt, cate_feat_tgt, marginal_repr_tgt], axis=-1)
    walks_bgd_new = np.concatenate([node_records_bgd, eidx_records_bgd, t_records_bgd, cate_feat_bgd, marginal_repr_bgd], axis=-1)
    return walks_src_new, walks_tgt_new, walks_bgd_new


def categorical_feat(walks_src, walks_tgt, walks_bgd, walks_src_new, walks_tgt_new, walks_bgd_new):
    node_records_src, eidx_records_src, t_records_src, out_anony_src = walks_src[:, :, :6], walks_src[:, :, 6:9], walks_src[:, :,9:12], walks_src[:, :,12:15]
    node_records_tgt, eidx_records_tgt, t_records_tgt, out_anony_tgt = walks_tgt[:, :, :6], walks_tgt[:, :, 6:9], walks_tgt[:, :, 9:12], walks_tgt[:, :, 12:15]
    node_records_bgd, eidx_records_bgd, t_records_bgd, out_anony_bgd = walks_bgd[:, :, :6], walks_bgd[:, :, 6:9], walks_bgd[:, :, 9:12], walks_bgd[:, :, 12:15]
    margin_src = walks_src_new[:,:,12:13]
    margin_tgt = walks_tgt_new[:,:,12:13]
    margin_bgd = walks_bgd_new[:,:,12:13]

    out_anony_src = out_anony_src.astype(int)
    out_anony_tgt = out_anony_tgt.astype(int)
    out_anony_bgd = out_anony_bgd.astype(int)
    num_data = out_anony_src.shape[0]
    num_walk_per_data = out_anony_src.shape[1]
    cate_feat_src = np.empty((num_data, num_walk_per_data, 1))
    cate_feat_tgt = np.empty((num_data, num_walk_per_data, 1))
    cate_feat_bgd = np.empty((num_data, num_walk_per_data, 1))

    strint_rep = {}
    t = 0
    for item in ["1,2,1", "1,2,2", "1,2,3", "1,2,0", "1,3,1", "1,3,3", "1,3,2", "1,3,0", "1,1,3", "1,1,2", "1,1,1",
                 "1,1,0"]:
        array_rep = np.array(list(eval(item)))
        string = np.array2string(array_rep)
        strint_rep[string] = t
        t = t + 1

    for i in tqdm(range(num_data)):
        samples_src = out_anony_src[i]  # [N**2, 3]
        samples_tgt = out_anony_tgt[i]  # [N**2, 3]
        samples_bgd = out_anony_bgd[i]  # [N**2, 3]
        for t in range(samples_src.shape[0]):
            anony_string = np.array2string(samples_src[t])
            cate_feat_src[i,t] = strint_rep[anony_string]
            anony_string = np.array2string(samples_tgt[t])
            cate_feat_tgt[i,t] = strint_rep[anony_string]
            anony_string = np.array2string(samples_bgd[t])
            cate_feat_bgd[i,t] = strint_rep[anony_string]

    walks_src_new = np.concatenate([node_records_src, eidx_records_src, t_records_src, cate_feat_src, margin_src], axis=-1)
    walks_tgt_new = np.concatenate([node_records_tgt, eidx_records_tgt, t_records_tgt, cate_feat_tgt, margin_tgt], axis=-1)
    walks_bgd_new = np.concatenate([node_records_bgd, eidx_records_bgd, t_records_bgd, cate_feat_bgd, margin_bgd], axis=-1)
    return walks_src_new, walks_tgt_new, walks_bgd_new


import concurrent.futures

def process_batch(batch_idx, edge_ids):
    bsz, m, c = edge_ids.shape[0], edge_ids.shape[1], edge_ids.shape[2]
    size1 = batch_idx.shape[0]
    edge_bsz = edge_ids[batch_idx].reshape([size1, m, c]) #[bsz, m, c]
    z = np.zeros(tuple(list(edge_bsz.shape) + [edge_bsz.max() + 1]))
    z[tuple(list(np.indices(z.shape[:-1])) + [edge_bsz])] = 1  # one-hot emb [bsz, m, c, n_max]
    count = z.sum(1).transpose([0, 2, 1])  # [bsz,  n_max, c]
    idx = edge_bsz.reshape(size1, m * c)
    bsz_idx = np.indices(idx.shape)[0]
    feat = count[bsz_idx, idx].reshape([size1, m, c, c])
    return feat

def edge_info(edge_ids):
    '''
    :param edge_ids: [bsz, n_walks, length]
    :return: [bsz, n_walks, length, length]
    '''
    bsz, m, c = edge_ids.shape[0], edge_ids.shape[1], edge_ids.shape[2]
    emb = np.zeros(tuple(list(edge_ids.shape) + [edge_ids.shape[-1]]))  # [bsz, m, c, c]
    batchsize = 100
    num_batches = math.ceil(bsz / batchsize)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for k in range(num_batches):
            s_id = k * batchsize
            e_id = min(bsz - 1, s_id + batchsize)
            if s_id == e_id:
                continue
            batch_idx = np.arange(s_id, e_id)
            futures.append(executor.submit(process_batch, batch_idx, edge_ids))
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            batch_idx = np.arange(i*batchsize, min(bsz-1, (i+1)*batchsize))
            emb[batch_idx] = future.result()
    return emb


def _edge_info(edge_ids):
    '''
    :param edge_ids: [bsz, n_walks, length]
    :return: [bsz, n_walks, length, length]
    '''
    batchsize = 100
    bsz, m, c = edge_ids.shape[0], edge_ids.shape[1], edge_ids.shape[2]
    emb = np.zeros(tuple(list(edge_ids.shape) + [edge_ids.shape[-1]]))  # [bsz, m, c, c]
    num_batch = math.ceil(bsz / batchsize)
    idx_list = np.arange(bsz)
    for k in tqdm(range(num_batch)):
        s_id = k*batchsize
        e_id = min(bsz - 1, s_id + batchsize)
        if s_id == e_id:
            continue
        batch_idx = idx_list[s_id:e_id]
        size1 = batch_idx.shape[0]

        edge_bsz = edge_ids[batch_idx].reshape([size1,m,c]) #[bsz, m, c]
        z = np.zeros(tuple(list(edge_bsz.shape) + [edge_bsz.max() + 1]))
        z[tuple(list(np.indices(z.shape[:-1])) + [edge_bsz])] = 1  # one-hot emb [bsz, m, c, n_max]
        count = z.sum(1).transpose([0, 2, 1])  # [bsz,  n_max, c]
        idx = edge_bsz.reshape(size1, m * c)
        bsz_idx = np.indices(idx.shape)[0]
        feat = count[bsz_idx, idx].reshape([size1, m, c, c])
        emb[batch_idx] = feat
    return emb

def new_edge_info(edge_ids):
    '''
    :param edge_ids: [bsz, n_walks, length]
    :return: [bsz, n_walks, length, length]
    '''
    bsz, m, c = edge_ids.shape[0], edge_ids.shape[1], edge_ids.shape[2]
    emb = np.zeros(tuple(list(edge_ids.shape) + [edge_ids.shape[-1]]))  # [bsz, m, c, c]
    for k in tqdm(range(bsz)):
        cc = edge_ids[k]
        u, indices = np.unique(cc, return_inverse=True)
        temp_count = np.zeros(tuple([u.shape[0], cc.shape[-1]]))
        for i, item in enumerate(u):
            temp_count[i] = np.count_nonzero(cc == item, axis=0)
        idx = indices.reshape(m * c)
        bsz_feat = temp_count[idx].reshape([m, c, c])
        emb[k] = bsz_feat
    return emb


def calculate_edge(walks_src, walks_tgt, walks_bgd):
    node_records, eidx_records_src, t_records, cat_feat, marginal = walks_src[:, :, :6], walks_src[:, :, 6:9], walks_src[:, :, 9:12], walks_src[:, :, 12:13], walks_src[:,:,13:14]

    node_records, eidx_records_tgt, t_records,cat_feat, marginal = walks_tgt[:, :, :6], walks_tgt[:, :, 6:9], walks_tgt[:, :, 9:12], walks_tgt[:, :, 12:13], walks_tgt[:,:,13:14]

    node_records, eidx_records_bgd, t_records,cat_feat, marginal = walks_bgd[:, :, :6], walks_bgd[:, :, 6:9], walks_bgd[:, :, 9:12], walks_bgd[:, :, 12:13], walks_bgd[:,:,13:14]
    edge_src = new_edge_info(eidx_records_src.astype(int))
    edge_tgt = new_edge_info(eidx_records_tgt.astype(int))
    edge_bgd = new_edge_info(eidx_records_bgd.astype(int))
    edge_load = np.stack([edge_src, edge_tgt, edge_bgd],axis=0)
    return edge_load



# data_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'processed',
#                         f'{data}_{MODE}_cat.h5')
# file = h5py.File(data_path,'r')
# walks_src = file["walks_src_new"][:]
# walks_tgt = file["walks_tgt_new"][:]
# walks_bgd = file["walks_bgd_new"][:]
# file.close()
# print("start edge_features")
# edge_load = calculate_edge(walks_src, walks_tgt, walks_bgd)
# save_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'processed',
#                         f"{data}_{MODE}_edge.npy")
# np.save(save_path, edge_load)
# print(f"Done {data} {MODE}")



### Model initialize
# for MODE in [ "train"]:
#     for data in ["wikipedia"]:
#         print(f"start {data} and {MODE}")
#         gnn_model_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'params', 'tgnn', f'{model_name}_{data}.pt')
#         tgat = torch.load(gnn_model_path)
#         tgat = tgat.to(device)
#
#         rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l, finder = load_data(mode=MODE, data=data)
#         tgat.ngh_finder = finder
#         pre_processing(tgat, rand_sampler, test_src_l, test_dst_l, test_ts_l, test_e_idx_l, MODE=MODE, data=data)
#
#         file = h5py.File(f'{data}_{MODE}.h5','r')
#         subgraph_src_0 = file["subgraph_src_0"][:]
#         subgraph_src_1 = file["subgraph_src_1"][:]
#         subgraph_tgt_0 = file["subgraph_tgt_0"][:]
#         subgraph_tgt_1 = file["subgraph_tgt_1"][:]
#         subgraph_bgd_0 = file["subgraph_bgd_0"][:]
#         subgraph_bgd_1 = file["subgraph_bgd_1"][:]
#         walks_src = file["walks_src"][:]
#         walks_tgt = file["walks_tgt"][:]
#         walks_bgd = file["walks_bgd"][:]
#         dst_fake = file["dst_fake"][:]
#         file.close()
#
#         walks_src_new, walks_tgt_new, walks_bgd_new = marginal(walks_src, walks_tgt, walks_bgd)
#         file_new = h5py.File(f"{data}_{MODE}_cat.h5", "w")
#         file_new.create_dataset("subgraph_src_0", data=subgraph_src_0)
#         file_new.create_dataset("subgraph_src_1", data=subgraph_src_1)
#         file_new.create_dataset("subgraph_tgt_0", data=subgraph_tgt_0)
#         file_new.create_dataset("subgraph_tgt_1", data=subgraph_tgt_1)
#         file_new.create_dataset("subgraph_bgd_0", data=subgraph_bgd_0)
#         file_new.create_dataset("subgraph_bgd_1", data=subgraph_bgd_1)
#         file_new.create_dataset("walks_src_new", data=walks_src_new)
#         file_new.create_dataset("walks_tgt_new", data=walks_tgt_new)
#         file_new.create_dataset("walks_bgd_new", data=walks_bgd_new)
#         file_new.create_dataset("dst_fake", data=dst_fake)
#         file_new.close()
#         print(f"Done {data} {MODE}")
#
#

