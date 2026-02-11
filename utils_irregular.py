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
from torch_geometric.transforms import ToUndirected, RemoveIsolatedNodes

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

def create_conflict_graph(graph, target_dual_degree):
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
    fixed_edges = []
    for tuple_pair in dual_edges:
        new_tuple_pair = []
        for tup in tuple_pair:
            st = sort_tuple(tup)
            transformed_value = tuples_as_numbers[st]
            new_tuple_pair.append(transformed_value)
        u, v = new_tuple_pair[0], new_tuple_pair[1]
        fixed_edges.append((u, v))
    if len(fixed_edges) > 0:
        edges = torch.tensor(fixed_edges, dtype=torch.long).t().contiguous()
    else:
        edges = torch.empty((2, 0), dtype=torch.long)
    dual_data = Data(edge_index=edges)
    transform = ToUndirected()
    dual_data = transform(dual_data)
    last_transform = RemoveIsolatedNodes()
    dual_data = last_transform(dual_data)
    # update the adj matrix or number of nodes after final transform
    K = dual_data.num_nodes
    edge_index = dual_data.edge_index
    A = torch.zeros((K, K), dtype=torch.float)
    A[edge_index[0], edge_index[1]] = 1
    degrees = A.sum(dim=1).to(dtype=torch.long)
    return dual_data, A, K, dual_data.edge_index.shape[1], degrees

##########################################################
##########################################################
##########################################################
##########################################################
################## other function ########################
##########################################################
##########################################################
##########################################################
##########################################################

def create_conflict_graph_dataset(graph_dataset, batch_size=32, return_data_for_plotting=False):
    '''
    This function will receive the dataset created (or uploaded from a certain file in the main.py)
    and return a dataset that contains the conflict graph for each timestep of the initial
    communication graph. That is, for each initial_graph consisting of nodes representing
    users and then edges representing links, this function will create conflict_graph with
    nodes that represent each link in initial_graph and edges that represent interference
    between said links.
    '''

    ### TODO adapt this function to return SAWLData objects, including A matrices and such.
   
    loader = graph_dataset
    conflict_data_train = []
    conflict_data_test = []

    for phase in loader:
        # For each phase (train, test)
       for data, batch in loader[phase]:
            # That is, for each batch
            samples = len(data)
            for i in range(samples):
                # For each graph in the batch

                # We define a graph using network X that has the same edges as ours,
                # then compute its dual graph and turn the resulting graph into a Data
                # object (and place it back into a Pytorch Dataset).

                initial_edges = data[i].edge_index
                initial_edges = initial_edges.t().tolist()
                #print('aristas grafo original ',initial_edges)
                G = nx.Graph()
                G.add_edges_from(initial_edges)

                # We create the corresponding conflict graph
                dual_data = create_conflict_graph(G)
                
                if phase == 'train':
                    conflict_data_train.append(dual_data)
                else:
                    conflict_data_test.append(dual_data)
                     
    conflict_loader = {}
    # We define the loader so that it has a dataset for each phase (train, test)
    
    conflict_dataset_train = SAWLDataset(conflict_data_train)
    conflict_loader['train'] = DataLoader(dataset=conflict_dataset_train, batch_size=batch_size)
    conflict_dataset_test = SAWLDataset(conflict_data_test)
    conflict_loader['test'] = DataLoader(dataset=conflict_dataset_test, batch_size=batch_size)

    # Check if data for plotting is needed (mainly for testing the code)
    if return_data_for_plotting:
        return conflict_loader, conflict_data_train
    else:
        return conflict_loader
    
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

def sample_lambda(epoch, T, option, scale, device, K, dist_lim, i, observed_lambdas=[]):
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
    elif len(observed_lambdas) != 0: # The remaining 55% of time I sample lambda from values seen in the previous epoch.
        timestep = np.random.randint(0, T)
        # observed_lambdas_values = np.concatenate([np.array(vec) for vec in observed_lambdas])
        # idx = np.random.randint(0, len(observed_lambdas_values), K)
    #     lambda_dual = torch.tensor(observed_lambdas_values[idx], dtype=torch.float).to(device)
    # return lambda_dual
        lambda_dual = torch.tensor(observed_lambdas[i][timestep], dtype=torch.float).to(device)
    return lambda_dual

def L_check(epoch, batch_size, scheduling_decisions, conflicts, phase, results, lambdas, K, Delta):
    if epoch%1 == 0:
        Lag = 0
        for i in range(batch_size):
            first_term_Lagrangian = torch.matmul(torch.mean(scheduling_decisions, dim=0), conflicts.T)
            second_term_Lagrangian = torch.matmul(lambdas[i].view(K), (torch.mean(scheduling_decisions, dim=0)*conflicts-Delta).view(K))
            Lag = Lag + first_term_Lagrangian + second_term_Lagrangian
        Lag_avg = Lag/batch_size
        results[phase, 'L_avg'].append(Lag_avg.detach().cpu().numpy())
    return(results)

def gradient_check(model):
    params = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    pgrad_norm = np.sqrt(np.sum([p.grad.norm().item()**2 for p in params]))
    return(pgrad_norm)

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