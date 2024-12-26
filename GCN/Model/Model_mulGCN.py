import torch
import torch.nn as nn
from Layer import GCN, MLP
from torch_geometric.nn import global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep, AttentionalAggregation
from torch_geometric.nn.norm import GraphNorm
from torch.nn import Linear
import torch.nn.functional as F

#Model architecture
'''
num_features_pro: the input dimension of the model
inner_dim: inner dimension of multi layer perceptron
output_dim: output dimension of graph convolutional layers
num_layers: number of MLP layers
All these parameters are adjustable
'''

class staGCNN(nn.Module):
    def __init__(self, dropout=0.1,
                 n_output=1,
                 num_features_pro=1024,
                 inner_dim=512,
                 output_dim=2048,
                 num_layers=2):
        super(staGCNN, self).__init__()

        print('Model Loaded')

        # For wild_type
        self.n_output = n_output
        self.wild_conv1 = GCN(num_features_pro, num_features_pro)
        self.wild_conv2 = GCN(num_features_pro, num_features_pro)
        self.wild_fc1 = nn.Linear(num_features_pro, output_dim)

        # For mutant
        self.mutant_conv1 = GCN(num_features_pro, num_features_pro)
        self.mutant_conv2 = GCN(num_features_pro, num_features_pro)
        self.mutant_fc1 = nn.Linear(num_features_pro, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Combined layers
        self.wild_MLP = MLP(2*output_dim, n_output, inner_dim, num_layers, dropout)

    def forward(self, wild_data, mutant_data):
        # Get graph input for wild_type
        wild_x, wild_edge_index, wild_batch = wild_data.x, wild_data.edge_index, wild_data.batch
        gcnn_concat = []
        gcnn_mut_concat = []
        mutant_x, mutant_edge_index, mutant_batch = mutant_data.x, mutant_data.edge_index, mutant_data.batch

        h = wild_x
        x = self.wild_conv1(wild_x, wild_edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        h = h + x
        x = self.wild_conv2(h, wild_edge_index)

        # Global pooling
        x = gmp(x, wild_batch)

        x = self.relu(self.wild_fc1(x))
        x = self.dropout(x)

        # Get graph input for mutant
        ht = mutant_x
        xt = self.mutant_conv1(mutant_x, mutant_edge_index)
        xt = self.relu(xt)
        xt = self.dropout(xt)
        ht = ht + xt
        xt = self.mutant_conv2(ht, mutant_edge_index)

        # Global pooling
        xt = gmp(xt, mutant_batch)

        xt = self.relu(self.mutant_fc1(xt))
        xt = self.dropout(xt)

        # Concatenation  
        xc = torch.cat((xt, x),1)

        # Dense layers
        out = self.wild_MLP(xc)

        return out
