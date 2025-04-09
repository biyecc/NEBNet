import os
import sys
from collections import namedtuple
from tqdm import tqdm

import numpy as np
import torch
import torch_geometric
import dgl

sys.path.insert(0, "../")
import argparse

from v1.dataset import TxPDataset
from v1.main import KFOLD

parser = argparse.ArgumentParser()
parser.add_argument("--savename", default="edge_node_graphs", type=str) ##############
parser.add_argument("--size", default=256, type=int)
parser.add_argument("--emb_path", default="features", type=str) ##############
parser.add_argument("--data", default="/data/CC/EGN-main/10xgenomics", type=str) ##############

args = parser.parse_args()

def get_edge_edge(edge_index):
    edge_edge_index = []
    _, new_sort_index = torch.sort(edge_index, descending=False, dim=1)  # 已经是第一行小的情况了
    edge_num = edge_index.shape[1]

    rank2ori = new_sort_index[0, :]
    cur_edges = edge_index[:, rank2ori]
    last_node = -1
    for each_edge in range(edge_num):
        in_node = cur_edges[0, each_edge]
        if in_node==last_node:
            edge_edge_index.append(torch.Tensor([rank2ori[each_edge-1], rank2ori[each_edge]]))
        else:
            last_node=in_node

    rank2ori = new_sort_index[1, :]
    cur_edges = edge_index[:, rank2ori]
    last_node = -1
    for each_edge in range(edge_num):
        in_node = cur_edges[0, each_edge]
        if in_node == last_node:
            edge_edge_index.append(torch.Tensor([rank2ori[each_edge - 1], rank2ori[each_edge]]))
        else:
            last_node = in_node
    edge_edge_index = torch.vstack(edge_edge_index).T
    return edge_edge_index

def get_split(idx, *ss):
    for i in range(len(ss)-1):
        if idx>=ss[i] and idx<ss[i+1]:
            return i
    return len(ss)-1

def remove_duplicate_edges(edge_index):
    # 确保每条边的第一个节点索引小于第二个节点索引
    sorted_edge_index = torch.sort(edge_index, dim=0)[0]

    # 转换为无向边，并去除重复
    undirected_edge_index = sorted_edge_index.numpy()
    return torch.from_numpy(np.unique(undirected_edge_index, axis=1))

def get_edge(x, percent=0.01):
    x_ = torch.tensor(x)
    adjs = []

    for each in tqdm(x):
        adj = torch.norm(each - x, dim=2, p=2)
        adjs.append(adj.squeeze())
    adjs = torch.vstack(adjs)
    threshold = torch.quantile(adjs[:, adjs.shape[1]//2], percent)
    adjs = adjs<threshold
    # 方法一：该方法较为准确
    edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(adjs)

    return edge_index

for fold in [0, 1, 2]:
    print(f"______开始处理fold{fold}______")
    savename = args.savename + "/" + str(fold)
    os.makedirs(savename, exist_ok=True)

    temp_arg = namedtuple("arg", ["size", "emb_path", "data"])
    temp_arg = temp_arg(args.size, args.emb_path, args.data)
    train_dataset = TxPDataset(KFOLD[fold][0], None, None, temp_arg, train=True)

    temp_arg = namedtuple("arg", ["size", "emb_path", "data"])
    temp_arg = temp_arg(args.size, args.emb_path, args.data)
    foldername = f"{savename}"
    os.makedirs(foldername, exist_ok=True)

    for iid in range(len(KFOLD[fold][0]) + len(KFOLD[fold][1])):
        dataset = TxPDataset([iid], None, None, temp_arg, train=False)

        dataset.min = train_dataset.min.clone()
        dataset.max = train_dataset.max.clone()

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1
        )
        img_data = []
        for x in loader:
            pos, p, py = x["pos"], x["p_feature"], x["count"]
            img_data.append([pos, p, py])

        data = torch_geometric.data.HeteroData()
        # 点属性相关
        data["window"].pos = torch.cat(([i[0] for i in img_data])).clone()
        data["window"].x = torch.cat(([i[1] for i in img_data])).clone()
        data["window"].x = data["window"].x.squeeze()
        data["window"].y = torch.cat(([i[2] for i in img_data])).clone()

        assert len(data["window"]["pos"]) == len(data["window"]["x"]) == len(data["window"]["y"])

        percent=0.0003
        window_edge = get_edge(torch.cat(([i[1] for i in img_data])).clone(), percent=percent)
        window_edge = remove_duplicate_edges(window_edge)
        print(f"{percent}时，获得的index shape为：", window_edge.shape)
        data['window', 'near', 'window'].edge_index = window_edge
        pos_edge_index = torch_geometric.nn.knn_graph(data["window"]["pos"], k=3, loop=False)
        pos_edge_index = remove_duplicate_edges(pos_edge_index)
        data["window", "close", "window"].edge_index = pos_edge_index
        edge_index = torch_geometric.nn.knn_graph(data["window"]["x"], k=3, loop=False)
        edge_index = remove_duplicate_edges(edge_index)
        data["window", "sim", "window"].edge_index = edge_index

        # all_edges边属性、ij2idx、和edge的边
        all_edges = torch.concat([window_edge, pos_edge_index, edge_index], dim=-1)
        edge_edge_index = get_edge_edge(all_edges)  # 2023 11 20 调试至此
        we, pei, ei = len(window_edge), len(pos_edge_index), len(edge_index)
        ij2idx = {}
        edge_features = []
        for i, edge in enumerate(all_edges.T):
            ij2idx[tuple(edge.tolist())] = i
            r_one_hot_i = get_split(i, [0, we, we+pei, we+pei+ei])
            r_one_hot = torch.zeros(3)
            r_one_hot[r_one_hot_i] = 1
            edge_features.append(
                torch.cat([data["window"].x[edge[0]], data["window"].x[edge[1]], r_one_hot, torch.tensor([edge[0]-edge[1]]),
                              torch.tensor([torch.norm((data["window"].pos[edge[0]]-data["window"].pos[edge[1]]).float(), p=2)])])
            )
        edge_features = torch.vstack(edge_features)
        data['edge'].x = edge_features
        data['edge', 'edge_edge', 'edge'].edge_index = edge_edge_index
        data['edge'].ij2idx = ij2idx
        torch.save(data, f"{foldername}/{iid}.pt")



