# imports
import torch
import numpy as np
from tqdm import tqdm
import os
import ast
import random
from collections import defaultdict
import argparse
import torch.optim as optim
import wandb
import sys

from gnn import GNN1
from utils import get_data, plot_lambdas, sample_lambda
from torch_geometric.loader import DataLoader

def sawl(num_epochs, batch_size, lr_lambda, lr_phi, weight_decay, delta, batch_norm, T0, objective, deg_lim, num_layers, num_features, dist_lim, scale):
    # initialize variables
    T = 200 # number of time slots
    T_0 = T0 # size of iteration window for averaging recent rates for dual variable updates 
            # (because during testing we use m instead of t)
    num_features_list = [1] + [num_features]*num_layers 
    lambda_granularity = 50
    
    # set computation device 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("the current device is ", device)

    # create datasets and dataloaders
    dataset_name = 'sawl_dataset_degree{}_graphs_60.pt'.format(deg_lim)
    print('dataset ', dataset_name)
    data = torch.load(os.path.join('data', dataset_name))
    train_data = data['train']
    test_data = data['test']
    loader = {}
    loader['train'] = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    loader['test'] = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
    # create model
    model = GNN1(num_features_list, batch_norm).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_phi)

    exp_name = 'exp_epochs_{}_delta_{}_lr_phi_{}_lr_lambda_{}_weight_decay{}_degree_{}_of_{}'.format(
                num_epochs, delta, lr_phi, lr_lambda, weight_decay, deg_lim, objective)
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    results = defaultdict(list)
    
    delt = delta
    for epoch in tqdm(range(num_epochs)):
        for phase in loader:
            # training
            if phase == 'train':
                print("train")
                model.train()
                L = []
                L_first_term = []
                L_second_term = []
                for i, data in enumerate(loader['train']):
                    data = data[0]
                    model.zero_grad()
                    optimizer.zero_grad()
                    data = data.to(device)
                    
                    # get relevant fields from data
                    edge_index, K, A, Delta = get_data(data, batch_size)
                    ones_vector = torch.ones(1, K, dtype=torch.float32).to(device)
                    for j in range(10):
                        with torch.set_grad_enabled(True):
                            option = np.random.rand(1)     
                            lambda_dual = sample_lambda(epoch, T, option, scale, device, K, dist_lim)
                            lambda_dual_gnn = lambda_dual.repeat(batch_size, 1).to(device) # the GNN requires this shape,
                                                                            # so we repeat for each batch
                            scheduling_decisions = model(lambda_dual_gnn.detach(), edge_index)
                            s = scheduling_decisions.view(K)
                        
                            # compute lagrangian and gradients
                            conflicts = ones_vector - torch.matmul(A, s)
                            conflicts = torch.clamp(conflicts, min=0)
                            
                            first_term_Lagrangian = objective*torch.matmul(s, conflicts.T)
                            second_term_Lagrangian = torch.matmul(lambda_dual.view(K), (s*conflicts).T)
                            Lagrangian = -(first_term_Lagrangian + second_term_Lagrangian) # negative because of adam optim
                            Lagrangian.backward()
                            
                            # update GNN parameters and zero the gradients
                            if optimizer is not None:
                                optimizer.step()
                                optimizer.zero_grad()
                                    
                    L.append(-Lagrangian.detach().cpu().numpy())  
                    L_first_term.append(first_term_Lagrangian.detach().cpu().numpy())
                    L_second_term.append(second_term_Lagrangian.detach().cpu().numpy())
                # save results from training                
                results[phase, 'L'].append(np.mean(np.array(L)))
                
            elif phase == 'test':
                # testing
                print("test")
                model.eval()
                of = []
                mc = []
                rc = []
                tx_sum = []
                succ_tx_sum = []
                
                for i, data in enumerate(loader['train']):
                    model.zero_grad()
                    data = data[0]
                    data.to(device)
                    
                    # get relevant fields from data
                    edge_index, K, A, Delta = get_data(data, batch_size)
                    ones_vector = torch.ones(1, K, dtype=torch.float32).to(device)

                    total_tx = torch.zeros(K).to(device)
                    succ_tx = torch.zeros(K).to(device)

                    with torch.set_grad_enabled(False):
                        all_decisions = []
                        obj_function = []
                        minimum_tx_constraint = []
                        dual_variable = []
                        lambdas_resilience = []

                        lambda_dual_test = torch.zeros(K, 1).to(device) #
                        lambda_dual_gnn_test = lambda_dual_test.repeat(batch_size, 1).to(device) # the GNN requires this shape,
                                                                        # so we repeat for each batch
                        for t in range(T):
                            scheduling_decisions = model(lambda_dual_gnn_test.detach(), edge_index).view(-1, K)
                            s = torch.mean(scheduling_decisions, dim=0)
                            s_binary = (s >= 0.5).float() # have 0s or 1s as values, instead of the continuous
                            
                            conflicts = ones_vector - torch.matmul(A, s)
                            conflicts = torch.clamp(conflicts, min=0)
                            conflicts_binary = ones_vector - torch.matmul(A, s_binary)
                            conflicts_binary = torch.clamp(conflicts_binary, min=0)

                            total_tx = total_tx + s_binary
                            succ_tx = succ_tx + s_binary*conflicts_binary

                            all_decisions.append(s_binary)
                            if (t+1)%T_0 == 0:
                                lambda_dual_test = lambda_dual_test - lr_lambda*((s*conflicts-delt).T
                                                                    +lambda_dual_test*weight_decay)
                                lambda_dual_test = torch.clamp(lambda_dual_test, min=0)
                                
                                lambda_dual_gnn_test = lambda_dual_test.repeat(batch_size, 1)  
                                lambda_dual_gnn_test = torch.clamp(lambda_dual_gnn_test, min=0).to(device) 

                            if weight_decay != 0 and t >= 0.95*T: # I average the last 5% lambdas
                                lambdas_resilience.append(lambda_dual_test)
                            obj_function.append(torch.matmul(s_binary, conflicts_binary.T))
                            minimum_tx_constraint.append((s_binary*conflicts_binary).view(K))

                            if epoch%lambda_granularity == 0 or epoch==num_epochs-1:
                                dual_variable.append(lambda_dual_test)
                            
                        tx_sum.append(torch.sum(total_tx).cpu())
                        succ_tx_sum.append(torch.sum(succ_tx).cpu())

                        minimum_tx_constraint = torch.stack(minimum_tx_constraint).view(T, -1, K)
                        minimum_tx_constraint = (torch.mean(minimum_tx_constraint, dim=0)).view(K) # average over time
                        obj_function = torch.stack(obj_function)
                        obj_function = torch.mean(obj_function, dim=0) # average over time
                        of.append(obj_function.detach().cpu().numpy()*100/K)
                        violation_mask = minimum_tx_constraint.detach().cpu().numpy() < delt
                        comparisons = (violation_mask).sum()
                        mc.append(comparisons*100/K)
                        if weight_decay != 0: # I average the last 5% lambdas:
                            lambdas_resilience = torch.stack(lambdas_resilience).view(int(T*0.05), -1, K)
                            lambdas_resilience = (torch.mean(lambdas_resilience, dim=0)).view(K) # average over time
                            resilient_constraint = (minimum_tx_constraint.detach().cpu().numpy() - delt + weight_decay*lambdas_resilience.detach().cpu().numpy() < 0).sum()
                            rc.append(resilient_constraint*100/K)
                            
                        if epoch%lambda_granularity == 0 or epoch==num_epochs-1:
                            dual_variable = torch.stack(dual_variable) # I want to keep all T values for all edges.
                
                results[phase, 'minimum_tx_constraint'].append(np.mean(np.array(mc)))
                results[phase, 'objective_function'].append(np.mean(np.array(of)))
                results[phase, 'transmissions'].append(np.mean(np.array(tx_sum)))
                results[phase, 'succ_transmissions'].append(np.mean(np.array(succ_tx_sum)))
                
                if weight_decay != 0:
                    results[phase, 'constraint_with_resilience'].append(np.mean(np.array(rc)))
                
                if epoch%lambda_granularity == 0 or epoch==num_epochs-1:
                    results[phase, 'lambda'].append(dual_variable.detach().cpu().numpy())

                if weight_decay != 0:
                    results[phase, 'constraint_with_resilience'].append(np.mean(np.array(rc)))
                
                if epoch%lambda_granularity == 0 or epoch==num_epochs-1:
                    results[phase, 'lambda'].append(dual_variable.detach().cpu().numpy())
                
        torch.save(results, './results/{}.json'.format(exp_name))
        torch.save(model.state_dict(), './models/{}.pt'.format(exp_name))
    wandb.finish()
    print('experiment finished')     

if __name__ == '__main__':
    def parse_float_list(list_str):
        value = ast.literal_eval(list_str)
        if not isinstance(value, (list, tuple)):
            raise ValueError
        return [float(x) for x in value]
        
    # set random seed
    random_seed = 32
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    print('~.~.~.~.~.~.~.~')
    print('random seed: ', random_seed)
    print("----------------------------")

    parser = argparse.ArgumentParser(description= 'System configuration')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr_lambda', type=float, default=2)
    parser.add_argument('--lr_phi', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--degree', type=int, default=6)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--norm', type=int, default=1)
    parser.add_argument('--num_features', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--T0', type=int, default=1)
    parser.add_argument('--obj_function', type=int, default=1)
    parser.add_argument('--distributions', type=parse_float_list, default=[0.3, 0.6, 1])
    
    args = parser.parse_args()
    print(f'epochs: {args.epochs}')
    print(f'batch_size: {args.batch_size}')
    print(f'lr_lambda: {args.lr_lambda}')
    print(f'lr_phi: {args.lr_phi}')
    print(f'delta: {args.delta}')
    
    sawl(num_epochs=args.epochs, batch_size=args.batch_size, lr_lambda=args.lr_lambda, 
         lr_phi=args.lr_phi, weight_decay=args.weight_decay,  delta=args.delta, deg_lim=args.degree, 
         num_features=args.num_features, objective=args.obj_function,
         batch_norm=args.norm, num_layers = args.num_layers,
         T0=args.T0, dist_lim=args.distributions, scale=args.scale)