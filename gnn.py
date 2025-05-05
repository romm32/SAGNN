import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LEConv, TAGConv, GINEConv, GATv2Conv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.nn.norm import BatchNorm, LayerNorm, GraphNorm, DiffGroupNorm
import copy
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class gnn_architecture_1(torch.nn.Module):
    def __init__(self, num_features_list, batch_norm):
        super(gnn_architecture_1, self).__init__()

        num_layers = len(num_features_list)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.batch_norm = batch_norm

        for i in range(num_layers - 1):
            if i < num_layers - 2:
                self.layers.append(TAGConv(num_features_list[i], num_features_list[i + 1], K=1, normalize=True, bias=True))
            else: # last layer
                self.layers.append(TAGConv(num_features_list[i], num_features_list[i + 1], K=1, normalize=True, bias=False))
            if self.batch_norm:
                self.norms.append(nn.BatchNorm1d(num_features_list[i+1]))
        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(param.data, mean=0.0, std=0.1)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0.1)


    def forward(self, y, edge_index):
        # In this function I should choose when/if we have biases, if 
        # we will have a non-linearity in all layers, which one it'll be
        for i, layer in enumerate(self.layers):
            y = layer(y, edge_index=edge_index)
            if self.batch_norm:
                y = self.norms[i](y)
            y = F.leaky_relu(y)
        return y
    
# main GNN module
class GNN1(torch.nn.Module):
    def __init__(self, num_features_list, batch_norm):
        super(GNN1, self).__init__()
        # I define a GNN with the previous architecture, then a linear
        # layer to have a one dimensional output from the model.
        self.gnn_backbone = gnn_architecture_1(num_features_list, batch_norm)
        self.b_p = nn.Linear(num_features_list[-1], 1, bias=False)
        
    def forward(self, y, edge_index):
        # I get the GNN output and then I apply a linear layer followed by a 
        # sigmoid function to get the probability of transmission for each node
        y = self.gnn_backbone(y, edge_index) 
        y = self.b_p(y)
        s = torch.sigmoid(y)
        return s