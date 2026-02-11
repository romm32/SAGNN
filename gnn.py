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
    

class NormLayer(nn.Module):
    def __init__(self, norm, in_channels, **kwargs):
        super().__init__()
        self.norm = norm
        self.in_channels = in_channels
        self.n_groups = kwargs.get('n_groups', 8)
        self.resolve_norm_layer(norm = self.norm, in_channels=in_channels, **kwargs)


    def resolve_norm_layer(self, norm, in_channels, **kwargs):
        n_groups = kwargs.get('n_groups', 8)

        if norm == 'batch':
            self.norm_layer = BatchNorm(in_channels)
        elif norm == 'layer':
            self.norm_layer = LayerNorm(in_channels)
        elif norm == 'graph':
            self.norm_layer = GraphNorm(in_channels=in_channels)
        elif norm == 'none' or norm is None:
            self.norm_layer = nn.Identity()


    def forward(self, x, batch = None, batch_size = None):
        if self.norm in ['batch', 'layer', 'group', 'none', None]:
            return self.norm_layer(x)
        elif self.norm in ['graph']:
            return self.norm_layer(x, batch = batch)
        else:
            print(self.norm)
            raise NotImplementedError
        

class ResGraphConvBlock(nn.Module):
    """ 
    Residual GCN block.
    """
    def __init__(self, conv_layer, norm_layer, activation, dropout_rate, res_connection, layer_ord = ['conv', 'norm', 'act', 'dropout']):
        super().__init__()
        self.conv_layer = conv_layer
        self.norm_layer = norm_layer
        self.activation = activation
        self.res_connection = res_connection
        self.dropout_rate = dropout_rate
        self.layer_ord = layer_ord


    def forward(self, y: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None):
        if self.layer_ord == ['conv', 'norm', 'act', 'dropout']:

            if any([isinstance(self.conv_layer, _) for _ in [LEConv, TAGConv]]):
                h = self.conv_layer(y, edge_index=edge_index)
            else:
                h = self.conv_layer(y, edge_index=edge_index)

            h = self.norm_layer(h, batch = batch) # identity layer if no batch/graph normalization is used.
            h = self.activation(h)
            h = F.dropout(h, p = self.dropout_rate, training=self.training)
            out = self.res_connection(y) + h

        else:
            raise NotImplementedError
        
        return out
    

def get_conv_model(conv_model_architecture, num_in, num_out, **kwargs):
    aggregation = kwargs.get('aggregation', None)
    k_hops = kwargs.get('k_hops', 2)
    num_heads = kwargs.get('num_heads', 2)
    dropout_rate = kwargs.get('dropout_rate', 0.0)

    if conv_model_architecture == 'LeConv':
        if aggregation is not None:
            conv_layer = LEConv(num_in, num_out, aggr = copy.deepcopy(aggregation))
        else:
            conv_layer = LEConv(num_in, num_out)

    elif conv_model_architecture == 'TAGConv':
        if aggregation is not None:
            conv_layer = TAGConv(in_channels=num_in, out_channels=num_out, K=k_hops, normalize=False, aggr = copy.deepcopy(aggregation))
        else:
            conv_layer = TAGConv(in_channels=num_in, out_channels=num_out, K=k_hops, normalize=False)
    else:
        raise NotImplementedError

    return conv_layer   

# residual backbone GNN class with layer ordering (Conv -> Norm -> Nonlinearity -> Dropout) + Residual connection
class res_gnn_backbone(torch.nn.Module):
    def __init__(self, conv_model_architecture, num_features_list, norm_layer_value, **kwargs):
        super(res_gnn_backbone, self).__init__()

        layer_ord = kwargs.get('layer_order', ['conv', 'norm', 'act', 'dropout'])
        self.layer_ord = layer_ord

        k_hops = kwargs.get('k_hops', 2)
        num_layers = len(num_features_list)
        activation = kwargs.get('activation', 'leaky_relu')
        aggregation = kwargs.get('aggregation', None)

        self.dropout_rate = kwargs.get('dropout_rate', 0.0)
        num_heads = kwargs.get('num_heads', 2)
        norm = norm_layer_value#kwargs.get('norm_layer', 'batch')
        global_pooling = kwargs.get('global_pooling', None)

        # Define activation functions
        if activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'softplus':
            self.activation = F.softplus
        else:
            raise NotImplementedError
        
        # Define optional pooling layers after the last conv + (batch norm + nonlinearity + dropout) layer
        self.global_pooling_layer = None
        if global_pooling is not None:
            if global_pooling == 'max':
                self.global_pooling_layer = global_max_pool
            elif global_pooling == 'mean':
                self.global_pooling_layer = global_mean_pool
            elif global_pooling == 'add':
                self.global_pooling_layer = global_add_pool
        
        self.blocks = nn.ModuleList()

        for i in range(num_layers - 1):
            conv_layer = get_conv_model(conv_model_architecture=conv_model_architecture,
                                        num_in=num_features_list[i],
                                        num_out=num_features_list[i+1],
                                        aggregation=aggregation,
                                        k_hops=k_hops,
                                        num_heads=num_heads,
                                        dropout_rate=self.dropout_rate
                                        )
            norm_layer = NormLayer(norm=norm,
                                   in_channels=num_features_list[i+1] if self.layer_ord.index('norm') > self.layer_ord.index('conv') else num_features_list[i])

            # If the number of input channels is not equal to the number of output channels we have to project the shortcut connection
            if num_features_list[i] != num_features_list[i+1]:
                res_connection = nn.Linear(in_features=num_features_list[i], out_features=num_features_list[i+1], bias = False)
            else:
                res_connection = nn.Identity()
                
            res_block = ResGraphConvBlock(conv_layer=conv_layer, norm_layer=norm_layer, activation=self.activation,
                                          dropout_rate=self.dropout_rate, res_connection=res_connection,
                                          layer_ord=self.layer_ord
                                          )
            self.blocks.append(res_block)
            
    def forward(self, y, edge_index, batch = None):
        for block in self.blocks:
            y = block(y=y, edge_index=edge_index, batch=batch)

        if self.global_pooling_layer is not None and batch is not None:
            y = self.global_pooling_layer(y, batch)
            
        return y
    
### main GNN module to learn state-augmented wireless policies. ###
class PrimalGNN(torch.nn.Module):
    def __init__(self, conv_model, num_features_list, norm_layer_value, temperature, **kwargs):
        super(PrimalGNN, self).__init__()
        
        self.lambdas_max = kwargs.get('lambdas_max', 1.0)
        self.conv_layer_normalize = kwargs.get("conv_layer_normalize", False)
        
        self.gnn_backbone = res_gnn_backbone(conv_model_architecture=conv_model, num_features_list=num_features_list, norm_layer_value=norm_layer_value, **kwargs)
        self.b_p = nn.Linear(num_features_list[-1], 1, bias=True)
        self.apply(self.init_weights)
        self.temperature = temperature
        self.n_iters_trained = 0
    
    def forward(self, y, edge_index, batch = None):

        if self.conv_layer_normalize:
            edge_index = gcn_norm(  
                    edge_index, y.size(0),
                    flow=self.gnn_backbone.blocks[0].conv_layer.flow,
                    dtype=y.dtype)
            
        y = self.gnn_backbone(y, edge_index, batch = batch) # derive node embeddings
        y = self.b_p(y)
        s = torch.sigmoid(y/self.temperature)
        self.n_iters_trained += 1
        
        return s
    

    def init_weights(self, m):
        """ Custom weight initialization. """
        if isinstance(m, pyg_nn.TAGConv):
            # print("m: ", m)
            for lin in m.lins:
                torch.nn.init.kaiming_uniform_(lin.weight, nonlinearity='leaky_relu')  
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)  # Bias is initialized to zero
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)  # Xavier init for final layers
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)


    @property
    def prototype_name(self):
        return f"PrimalGNN_{self.gnn.depth}_layers"
    
