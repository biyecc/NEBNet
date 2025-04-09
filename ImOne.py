import torch
import torch.nn as nn
import torch_geometric.nn as pyg
import numpy as np

from ImpossibleHeteroconv import HeteroConv
from impossible_conv import SAGEConv
from components.TransformerPooling import GraphMultisetTransformer


class SophisticatedModel(torch.nn.Module):
    def __init__(self, num_layers=4, mdim=512, edge_channel=4101, edge_embed=True, global_embed=True):
        super().__init__()
        # edge_channel需要一个合适的表示（这什么量纲啊？）
        hidden_channels = 512
        out_channels = 250
        input_channel = mdim
        self.num_layers = num_layers
        self.edge_embed, self.global_embed = edge_embed, global_embed
        self.pretransform_win = pyg.Linear(input_channel, hidden_channels, bias=False)
        self.post_plot_transform = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            pyg.Linear(hidden_channels, hidden_channels, bias=False),
            nn.LeakyReLU(0.2, True),
        )
        self.pretransform_edge = pyg.Linear(edge_channel, hidden_channels, bias=False)
        self.post_edge_transform = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            pyg.Linear(hidden_channels, hidden_channels, bias=False),
            nn.LeakyReLU(0.2, True),
        )
        self.leaklyrelu = nn.LeakyReLU(0.2)
        #####################
        # 点嵌入更新块儿
        self.plot_convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('window', 'near', 'window'): SAGEConv(hidden_channels, hidden_channels, edge_embed=edge_embed),
                ('window', 'close', 'window'): SAGEConv(hidden_channels, hidden_channels, edge_embed=edge_embed),
                ('window', 'sim', 'window'): SAGEConv(hidden_channels, hidden_channels, edge_embed=edge_embed),
                # ('window', 'sim', 'window'): pyg.SAGEConv((hidden_channels,hidden_channels), hidden_channels, hidden_channels, add_self_loops = False),
            }, aggr='mean')
            self.plot_convs.append(conv)

        # 边嵌入更新块儿
        self.edge_convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = pyg.SAGEConv(hidden_channels, hidden_channels)
            self.edge_convs.append(conv)
        if global_embed:
            self.poolers = nn.ModuleList()
            for _ in range(num_layers):
                self.poolers.append(GraphMultisetTransformer(hidden_channels, hidden_channels, hidden_channels, None,
                                                             pool_sequences=['GMPool_I']))  # 只取最后一个环节获得一个节点

        # self.pool = CSRA(hidden_channels)
        self.lin = pyg.Linear(hidden_channels, out_channels)

    def forward(self, x_dict):
        # 数据集要求：点属性，边属性，以及配套的ij2idx来取边属性
        # edge_index, edge_edge_index
        x_dict['window'].x = self.post_plot_transform(self.pretransform_win(x_dict['window'].x))
        x_dict['edge'].x = self.post_edge_transform(self.pretransform_edge(x_dict['edge'].x))
        only_plot_edges = {key:item for key, item in x_dict.edge_index_dict.items() if 'edge' not in key}
        for l in range(self.num_layers):
            global_vec = None
            if self.global_embed:
                global_vec = self.poolers[l](x_dict['window'].x, batch=None)
            x_dict['window'].x = self.plot_convs[l]({'window':x_dict['window'].x}, only_plot_edges, global_vec,
                                                    x_dict['edge'].x, x_dict['edge'].ij2idx)['window']  # 这地方要改，引入边的信息
            x_dict['edge'].x = self.edge_convs[l](x_dict['edge'].x,
                                                  torch.tensor(x_dict['edge', 'edge_edge', 'edge'].edge_index,
                                                               dtype=torch.int64))
            # x_dict = {key: self.leaklyrelu(x) for key, x in x_dict.items()}
            for key in ['edge', 'window']:
                x_dict[key].x = self.leaklyrelu(x_dict[key].x)

        # return self.lin(self.pool(x_dict, edge_index_dict))
        return self.lin(x_dict['window'].x)
