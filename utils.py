import torch
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from scipy.spatial import distance
import os
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected, degree as pyg_degree

import numpy as np
from collections import defaultdict
from tqdm import tqdm
from torch_geometric.loader import DataLoader
                #from torch.utils.data import TensorDataset, DataLoader be careful using 
                # the corresponding dataloader for pytorch geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import ToUndirected

##########################################################
##########################################################
##########################################################
##########################################################
################## other function ########################
##########################################################
##########################################################
##########################################################
##########################################################

class SAWLDataset(Dataset):
    # Define a basic dataset
    def __init__(self, data_list):
        super().__init__(None, None, None)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx], idx

##########################################################
##########################################################
##########################################################
##########################################################
################## other function ########################
##########################################################
##########################################################
##########################################################
##########################################################

class SAWLData(Data):
    def __init__(self,
                 x=None,
                 edge_index=None,
                 A=None,
                 Delta=None,
                 K=None):
        super().__init__()
        self.x = x
        self.edge_index = edge_index
        self.A = A
        self.Delta = Delta
        self.K = K # I add the number of nodes in the conflict graph (that is,
                   # the amount of edges in the inital graph) for simplicity

##########################################################
##########################################################
##########################################################
##########################################################
################## other function ########################
##########################################################
##########################################################
##########################################################
##########################################################

def sort_tuple(tup):
    '''
    This function gets a tuple called tup containing two integers
    and it will sort it. This is needed so that a node called (0,1) and
    one called (1, 0) are considered the same.
    '''
    sorted_tuple = sorted(tup)
    return tuple(sorted_tuple)

##########################################################
##########################################################
##########################################################
##########################################################
################## other function ########################
##########################################################
##########################################################
##########################################################
##########################################################
# ------------------------------------------------------------

def create_conflict_graph(graph):
    # Create a dual graph from the input graph
    dual_graph = nx.line_graph(graph)
    dual_edges = list(dual_graph.edges())
    np.random.shuffle(dual_edges)
    tuples_as_numbers = {}
    next_number = 0
    for tuple_pair in dual_edges:
        for tup in tuple_pair:
            st = sort_tuple(tup)
            if st not in tuples_as_numbers:
                tuples_as_numbers[st] = next_number
                next_number += 1
    K = len(list(dual_graph.nodes()))
    A = torch.zeros(K, K)
    degrees = torch.zeros(K, dtype=torch.long)
    fixed_edges = []
    for tuple_pair in dual_edges:
        new_tuple_pair = []
        for tup in tuple_pair:
            st = sort_tuple(tup)
            transformed_value = tuples_as_numbers[st]
            new_tuple_pair.append(transformed_value)
        u, v = new_tuple_pair[0], new_tuple_pair[1]
        A[u, v] = 1
        A[v, u] = 1
        degrees[u] += 1
        degrees[v] += 1
        fixed_edges.append((u, v))
    if len(fixed_edges) > 0:
        edges = torch.tensor(fixed_edges, dtype=torch.long).t().contiguous()
    else:
        edges = torch.empty((2, 0), dtype=torch.long)
    dual_data = Data(edge_index=edges)
    transform = ToUndirected()
    dual_data = transform(dual_data)
    return dual_data, A, K, edges.shape[1], degrees
    
def get_data(data, batch_size):
    edge_index = data.edge_index # edges
    K = data.K # nodes in conflict graph
    K = K[0].cpu().numpy()
    A = data.A # A matrix
    A = A.view(batch_size, -1, K)
    Delta = data.Delta # minimum transmission requirement
    Delta = Delta.view(-1, K)
    Delta = Delta[0].view(K)
    return(edge_index, K, A, Delta)

def sample_lambda(epoch, T, option, scale, device, K, dist_lim):
    if option < dist_lim[0] or epoch == 0: # 15% of time I do uniform distribution
        lambda_dual = scale*torch.rand(K, 1).to(device)
    elif dist_lim[0] <= option and option < dist_lim[1]: # 15% of time I do uniform with 30% of zeros
        rand_vals = scale * torch.rand(K, 1, device=device)
        mask = (torch.rand(K, 1, device=device) >= 0.3).float()
        lambda_dual = (rand_vals * mask).to(device)
    elif dist_lim[1] <= option and option < dist_lim[2]: # 15% of time I do uniform with 25% of ones
        rand_vals = scale * torch.rand(K, 1)
        mask = torch.rand(K, 1) < 0.25
        lambda_dual = torch.where(mask, torch.tensor(1.0), rand_vals).to(device)
    return lambda_dual

def plot_lambdas(epoch, delta, lr_phi, lr_lambda, dual_variable, exp_name, weight_decay):
    plt.figure(figsize=(16,9))
    plt.title('Evolution of dual variable in epoch={}'.format(epoch))
    plt.suptitle('Delta={}, lr_primal={}, lr_dual={}, weight_decay={}'.format(delta, lr_phi, lr_lambda, weight_decay), fontsize=14, fontweight='bold')
    plt.ylabel('lambda in epoch {}'.format(epoch))
    plt.xlabel('Time step')
    plt.plot(np.linalg.norm(dual_variable.detach().cpu().numpy(), axis=1))
    image_name = 'results_average' + exp_name + 'lambda' + str(epoch) + '.png'
    image_path = os.path.join('results/lambdas', image_name)
    plt.savefig(image_path)
    return