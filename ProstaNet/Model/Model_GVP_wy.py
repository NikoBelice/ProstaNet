import torch
import copy
import torch.nn as nn
from typing import List
from torch import Tensor
from Layer_GVP import SinGVP
from Layer import MLP, Linear



#Program architecture
class StaGVP(nn.Module):
    def __init__(self, node_in_dim, node_h_dim, edge_in_dim, edge_h_dim, drop_rate):
        super(StaGVP, self).__init__()

        print('Model Loaded')

        self.wild_conv1 = SinGVP(node_in_dim, node_h_dim, edge_in_dim, edge_h_dim, drop_rate)
        
        self.mutant_conv1 = SinGVP(node_in_dim, node_h_dim, edge_in_dim, edge_h_dim, drop_rate)

        node_h_dim = (
            node_h_dim[0] * 3, #3 here is the number of GVP layer, it is an adjustable parameter
            node_h_dim[1] * 3,
        )
        ns, _ = node_h_dim

       #Multi-head attention layers
        self.interactions = nn.ModuleList([
            nn.MultiheadAttention(ns, num_heads=1, batch_first=True, dropout=drop_rate) for _ in range(2)])  
        self.norm = nn.LayerNorm(ns)
        self.dropout = nn.Dropout(drop_rate)

        self.linear = nn.Linear(ns, 1, bias=False)

    def forward(self, wild_data, mutant_data):
        h_V_w = (wild_data.node_s, wild_data.node_v)
        h_E_w = (wild_data.edge_s, wild_data.edge_v)
        edge_index_w = wild_data.edge_index
        batch_w = wild_data.batch

        h_V_m = (mutant_data.node_s, mutant_data.node_v)
        h_E_m = (mutant_data.edge_s, mutant_data.edge_v)
        edge_index_m = mutant_data.edge_index
        batch_m = mutant_data.batch
        
        x = self.wild_conv1(h_V_w, edge_index_w, h_E_w, batch_w)
        xt = self.mutant_conv1(h_V_m, edge_index_m, h_E_m, batch_m)

        xs = torch.concat([x.unsqueeze(1), xt.unsqueeze(1)], dim=1)

        for i, layer in enumerate(self.interactions):
            xs_updated, _ = layer(xs, xs, xs)  # self attention
            xs += self.dropout(xs_updated)  # residual connection
            if i < len(self.interactions) - 1:
                xs = self.norm(xs)

        out = torch.sigmoid(self.linear(xs[:, 0] - xs[:, 1]))

        return out
