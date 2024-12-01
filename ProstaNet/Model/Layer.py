import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
    
class GeneralLayer(nn.Module):
    '''General wrapper for layers'''
    def __init__(self, name, dim_in, dim_out, p, **kwargs):
        super(GeneralLayer, self).__init__()
        self.layer = layer_dict[name](dim_in, dim_out, **kwargs)
        layer_post = []
        layer_post.append(nn.ReLU())
        layer_post.append(nn.Dropout(p))
        self.post_layer = nn.Sequential(*layer_post)

    def forward(self, batch_x):
        batch_x = self.layer(batch_x)
        batch_x = self.post_layer(batch_x)
        return batch_x
    
class GeneralMultiLayer(nn.Module):
    '''General wrapper for stack of layers'''
    def __init__(self, name, num_layers, dim_in, dim_out, dim_inner, p, **kwargs):
        super(GeneralMultiLayer, self).__init__()
        dim_inner = dim_in if dim_inner is None else dim_inner
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_inner
            d_out = dim_out if i == num_layers - 1 else dim_inner
            layer = GeneralLayer(name, d_in, d_out, p, **kwargs)
            self.add_module('Layer_{}'.format(i), layer)

    def forward(self, batch_x):
        for layer in self.children():
            batch_x = layer(batch_x)
        return batch_x

class MLP(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 dim_inner,
                 num_layers,
                 p,
                 **kwargs):

        super(MLP, self).__init__()
        dim_inner = dim_in if dim_inner is None else dim_inner
        layers = []
        if num_layers > 1:
            layers.append(
                GeneralMultiLayer('linear',
                                  num_layers - 1,
                                  dim_in,
                                  dim_inner,
                                  dim_inner,
                                  p))
            layers.append(Linear(dim_inner, dim_out))
            layers.append(nn.Sigmoid())
        else:
            layers.append(Linear(dim_in, dim_out))
            layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, batch_x):
        batch_x = self.model(batch_x)
        return batch_x    

class Linear(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(Linear, self).__init__()
        self.model = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, batch_x):
        batch_x = self.model(batch_x)
        return batch_x
    
layer_dict = {
    'linear': Linear,
    'mlp': MLP
}
