import os, math, random
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected, degree as pyg_degree
import networkx as nx
import random
import torch
import os
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RadiusGraph
import networkx as nx
from torch_geometric.utils import contains_isolated_nodes
from torch_geometric.transforms import RemoveIsolatedNodes
from utils import SAWLDataset, create_conflict_graph, SAWLData
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
import math

def create_conflict_dataset(num_samples,
                            min_num_nodes,
                            max_num_nodes,
                            r, deltavalue, dual_tol, batch_size,
                            target_K, target_dual_degree,
                            noise, return_data_for_plotting=False):
    
    data_train, data_test = [], []
    original_data_train, original_data_test = [], []
    A_matrix_train, A_matrix_test = [], []
    K_train, K_test = [], []
    conflict_data, A, K, len_edges, degrees, data_transformed = \
        [], [], [], [], [], []
    number_of_graphs = num_samples['train'] + num_samples['test']

    another_transform = RemoveIsolatedNodes()
    count = 0
    while count < number_of_graphs:
        # We initialize the world, create a grid depending on the number
        # of users and position each user randomly in one of the cells.
        num_nodes = np.random.randint(min_num_nodes, max_num_nodes + 1)
        world_size = math.sqrt(num_nodes)
        grid_dim  = math.ceil(math.sqrt(num_nodes))  
        grid_rows = grid_cols = grid_dim
        cell_w = world_size / grid_cols
        cell_h = world_size / grid_rows

        centers = np.array([[(j + .5) * cell_w, (i + .5) * cell_h]
                            for i in range(grid_rows) for j in range(grid_cols)])

        idx = np.random.choice(len(centers), size=num_nodes, replace=False)
        positions = centers[idx] + np.random.normal(scale=noise*world_size,
                                                    size=(num_nodes, 2))
        pos = torch.from_numpy(positions).float()
        x = torch.ones(num_nodes, 1, dtype=torch.long)

        # We connect nodes within a radius r*{distance between users}
        edge_index = torch.tensor([], dtype=torch.long)
        radius = r*world_size/grid_cols
        transform = RadiusGraph(radius)
        data_transformed_i = another_transform(transform(Data(x=x, edge_index=edge_index, pos=pos)))
        
        edges = data_transformed_i.edge_index
        edges = edges.t().tolist()
        print('orig edges', len(edges))
        G = nx.Graph()
        G.add_edges_from(edges)
        conflict_data_i, A_i, K_i, len_edges_i, degrees_i = create_conflict_graph(graph=G)
        print("K", K_i)
        if target_K is None or abs(K_i - target_K) <= 50:
            print("D", np.mean(degrees_i.float().numpy()))
            if target_dual_degree is None or abs(np.mean(degrees_i.float().numpy()) - target_dual_degree) <= dual_tol:
                conflict_data.append(another_transform(conflict_data_i))
                A.append(A_i)
                K.append(K_i)
                len_edges.append(len_edges_i)
                degrees.append(degrees_i)
                data_transformed.append(data_transformed_i)
                count += 1
                print(count)

    for i in range(number_of_graphs):
        if i < num_samples['train']:
            data_train.append(conflict_data[i % number_of_graphs])
            A_matrix_train.append(A[i % number_of_graphs])
            K_train.append(K[i % number_of_graphs])
            original_data_train.append(data_transformed[i % number_of_graphs])
        else:
            data_test.append(conflict_data[i % number_of_graphs])
            A_matrix_test.append(A[i % number_of_graphs])
            K_test.append(K[i % number_of_graphs])
            original_data_test.append(data_transformed[i % number_of_graphs])
    sawl_data_train = []
    for i in range(len(data_train)):
        K_val = K_train[i]
        A_val = A_matrix_train[i]
        x_graph = torch.ones(K_val, dtype=torch.long).view(-1, 1)
        edge_index_graph = data_train[i].edge_index
        Delta = torch.full((K_val, 1), deltavalue)
        sawl_data_train.append(SAWLData(x=x_graph, edge_index=edge_index_graph, A=A_val, Delta=Delta, K=K_val))
    sawl_data_test = []
    for j in range(len(data_test)):
        K_val = K_test[j]
        A_val = A_matrix_test[j]
        x_graph = torch.ones(K_val, dtype=torch.long).view(-1, 1)
        edge_index_graph = data_test[j].edge_index
        Delta = torch.full((K_val, 1), deltavalue)
        sawl_data_test.append(SAWLData(x=x_graph, edge_index=edge_index_graph, A=A_val, Delta=Delta, K=K_val))
    loader = {}
    dataset_train = SAWLDataset(sawl_data_train)
    loader['train'] = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    dataset_test = SAWLDataset(sawl_data_test)
    loader['test'] = DataLoader(dataset=dataset_test, batch_size=batch_size)
    dataset_to_save = {'train': sawl_data_train, 'test': sawl_data_test}
    os.makedirs('data', exist_ok=True)
    torch.save(dataset_to_save, os.path.join('data', 'sawl_dataset_degree'+str(target_dual_degree)+'_graphs_'+ str(number_of_graphs) +'.pt'))
    if return_data_for_plotting:
        return loader, sawl_data_train, sawl_data_test, original_data_train, A_val[0], degrees
    else:
        return loader, A, len_edges, degrees


def create_conflict_dataset_irregular(num_samples,
                            nodes,
                            r, deltavalue, dual_tol, batch_size,
                            target_K, target_dual_degree,
                            noise,
                            density=False, density_level=0,
                            return_data_for_plotting=False):
    
    data_train, data_test = [], []
    original_data_train, original_data_test = [], []
    A_matrix_train, A_matrix_test = [], []
    K_train, K_test = [], []
    conflict_data, A, K, len_edges, degrees, data_transformed = \
        [], [], [], [], [], []
    number_of_graphs = num_samples['train'] + num_samples['test']

    another_transform = RemoveIsolatedNodes()
    count = 0
    while count < number_of_graphs:
        if not density:
            world_size = np.random.choice(nodes)
            num_nodes = world_size**2
        else:
            world_size = nodes[0]
            if density_level == 1:
                num_nodes = world_size**2 - 50
                target_K = 500
            elif density_level == 2:
                num_nodes = world_size**2
                target_K = 700
            elif density_level == 3:
                num_nodes = world_size**2 + 50
                target_K = 900
            elif density_level == 4:
                num_nodes = world_size**2 + 100
                target_K = 1200
            elif density_level == 0:
                num_nodes = world_size**2 - 100
                target_K = 350
           
        grid_dim  = world_size
        grid_rows = grid_cols = grid_dim
        cell_w = world_size / grid_cols
        cell_h = world_size / grid_rows

        centers = np.array([[(j + .5) * cell_w, (i + .5) * cell_h]
                            for i in range(grid_rows) for j in range(grid_cols)])

        if not density or num_nodes < len(centers):
            idx = np.random.choice(len(centers), size=num_nodes, replace=False)
        else:
            idx_firsts = np.random.choice(len(centers), size=len(centers), replace=False)
            idx_lasts = np.random.choice(len(centers), size=num_nodes-len(centers), replace=True)
            idx = np.empty(num_nodes, dtype=int)
            idx[:len(centers)] = idx_firsts
            idx[len(centers):] = idx_lasts
        positions = centers[idx] + np.random.normal(scale=noise * world_size,
                                                    size=(num_nodes, 2))
        pos = torch.from_numpy(positions).float()
        x = torch.ones(num_nodes, 1, dtype=torch.long)

        edge_index = torch.tensor([], dtype=torch.long)
        radius = r*world_size / grid_cols
        transform = RadiusGraph(radius)
        data_transformed_i = another_transform(transform(Data(x=x, edge_index=edge_index, pos=pos)))
        
        #if not has_isolated_nodes:
        edges = data_transformed_i.edge_index
        edges = edges.t().tolist()
        G = nx.Graph()
        G.add_edges_from(edges)
        conflict_data_i, A_i, K_i, len_edges_i, degrees_i = create_conflict_graph(graph=G, target_dual_degree=target_dual_degree)
        if target_K is None or abs(K_i - target_K) <= 50:
            if target_dual_degree is None or abs(np.mean(degrees_i.float().numpy()) - target_dual_degree) <= dual_tol:
                G = to_networkx(conflict_data_i, to_undirected=True)        
                # Check if the graph is connected
                is_connected = nx.is_connected(G)
                if is_connected:
                    print(is_connected)
                    conflict_data.append(conflict_data_i)
                    A.append(A_i)
                    K.append(K_i)
                    len_edges.append(len_edges_i)
                    degrees.append(degrees_i)
                    data_transformed.append(data_transformed_i)
                    count += 1
                    print(count)

    for i in range(number_of_graphs):
        if i < num_samples['train']:
            data_train.append(conflict_data[i % number_of_graphs])
            A_matrix_train.append(A[i % number_of_graphs])
            K_train.append(K[i % number_of_graphs])
            original_data_train.append(data_transformed[i % number_of_graphs])
        else:
            data_test.append(conflict_data[i % number_of_graphs])
            A_matrix_test.append(A[i % number_of_graphs])
            K_test.append(K[i % number_of_graphs])
            original_data_test.append(data_transformed[i % number_of_graphs])
    sawl_data_train = []
    for i in range(len(data_train)):
        K_val = K_train[i]
        A_val = A_matrix_train[i]
        x_graph = torch.ones(K_val, dtype=torch.long).view(-1, 1)
        edge_index_graph = data_train[i].edge_index
        Delta = torch.full((K_val, 1), deltavalue)
        sawl_data_train.append(SAWLData(x=x_graph, edge_index=edge_index_graph, A=A_val, Delta=Delta, K=K_val))
    sawl_data_test = []
    for j in range(len(data_test)):
        K_val = K_test[j]
        A_val = A_matrix_test[j]
        x_graph = torch.ones(K_val, dtype=torch.long).view(-1, 1)
        edge_index_graph = data_test[j].edge_index
        Delta = torch.full((K_val, 1), deltavalue)
        sawl_data_test.append(SAWLData(x=x_graph, edge_index=edge_index_graph, A=A_val, Delta=Delta, K=K_val))
    loader = {}
    dataset_train = SAWLDataset(sawl_data_train)
    loader['train'] = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    dataset_test = SAWLDataset(sawl_data_test)
    loader['test'] = DataLoader(dataset=dataset_test, batch_size=batch_size)
    dataset_to_save = {'train': sawl_data_train, 'test': sawl_data_test}
    os.makedirs('data', exist_ok=True)
    torch.save(dataset_to_save, os.path.join('data', 'density_sawl_dataset_degree'+str(noise)+'_graphs_'+ str(number_of_graphs) +'.pt'))
    if return_data_for_plotting:
        return loader, sawl_data_train, sawl_data_test, original_data_train, A_val[0], degrees
    else:
        return loader, A, len_edges, degrees

