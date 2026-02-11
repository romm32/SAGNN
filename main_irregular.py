# imports
import ast
import torch
import numpy as np
from tqdm import tqdm
import os
import random
from collections import defaultdict
import argparse
import torch.optim as optim
from matplotlib import pyplot as plt
import sys
from torch_geometric.loader import DataLoader

from gnn import PrimalGNN
from utils_irregular import get_data, plot_lambdas, sample_lambda

def sawl(args, num_exps, num_epochs, batch_size, lr_lambda, lr_phi, weight_decay, delta, batch_norm, how_many_graphs, small_graph, objective, dist_lim, num_layers, num_features, hops=2, norm_layer='graph', deg_lim=5, scale=1, gnn_original=1, plot=False, added=0):

    # initialize variables
    T = 200 # number of time slots
    batch_size = batch_size
    num_features_list = [1] + [num_features]*num_layers # number of GNN features in layers 1, 2, 3, then
                                    # the last layer is introduced in the GNN definition,
                                    # according to the size of the output.
    num_epochs = num_epochs
    lr_lambda_use = lr_lambda
    lambda_granularity = 50
    delt = delta

    # set computation device 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("the current device is ", device)

    # create datasets and dataloaders
    print('loading data')
    print("----------------------------")
    dataset_name = 'sawl_dataset_degree6_graphs_15.pt'#'sawl_dataset_degree{}_graphs_150.pt'.format(deg_lim)
    print('dataset ', dataset_name)
    data = torch.load(os.path.join('data', dataset_name))
    train_data = data['train']
    test_data = data['test']
    loader = {}
    loader['train'] = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    loader['test'] = DataLoader(test_data, batch_size=batch_size)

    D = num_exps # experiment number
    for d in range(D):
        
        print('training and evaluating model ', d)
        print("----------------------------")
        
        # create model
        model = PrimalGNN(conv_model='TAGConv', num_features_list=num_features_list, norm_layer_value=norm_layer,
                              #scale=scale, 
                              temperature=10,
                              k_hops = hops, norm_layer = norm_layer).to(device)
        lr_phi_use = lr_phi
        optimizer = optim.Adam(model.parameters(), lr=lr_phi_use)

        exp_name = 'exp_{}_epochs_{}_delta_{}_lr_phi_{}_lr_lambda_{}_weight_decay{}_distributions{}_{}_{}_hops_{}_norm_layer_{}__how_many_graphs_{}_num_features_{}_num_layers_{}'.format(
                    d, num_epochs, delta, lr_phi, lr_lambda, weight_decay, dist_lim[0], dist_lim[1], dist_lim[2], hops, norm_layer, how_many_graphs, num_features, num_layers)
                    
        os.makedirs('results', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        results = defaultdict(list)
        distributions = []
        num_graphs_train = len(train_data)
        for x in range(num_graphs_train):
            distributions.append([])

        baseline = 0.0
        baseline_alpha = 0.5
        updates = 0
        penalty = 1
        for epoch in tqdm(range(num_epochs)):
            for phase in loader:
                # training
                if phase == 'train':
                    print("train")
                    model.train()
                    L = []
                    L_first_term = []
                    L_second_term = []
                    k = 0
                    for i, data in enumerate(loader['train']):
                        if k < how_many_graphs:
                            model.zero_grad()
                            optimizer.zero_grad()
                            data = data[0].to(device)
                            
                            # get relevant fields from data
                            edge_index, K, A, Delta = get_data(data, batch_size)
                            ones_vector = torch.ones(1, K, dtype=torch.float32).to(device)
                    
                            for j in range(10): # each graph should see a few lambdas
                                # this number may have to be smaller in order to the update
                                # rate of the gnn not to be so fast
                                with torch.set_grad_enabled(True):
                                    option = np.random.rand(1)     
                                    lambda_dual = sample_lambda(epoch, T, option, scale, device, K, dist_lim, distributions, i)#scale*torch.rand(K, 1).to(device)# 
                                    lambda_dual_gnn = lambda_dual.repeat(batch_size, 1).to(device) # the GNN requires this shape,
                                                                                    # so we repeat for each batch
                                    # sample the GNN 
                                    node_probs = model(lambda_dual_gnn.detach(), scale, edge_index).view(-1, K)
                                    s_distribution = torch.distributions.Bernoulli(node_probs)
                                    s = s_distribution.sample().view(K)

                                    # compute lagrangian and gradients
                                    conflicts = ones_vector - torch.matmul(A, s)
                                    conflicts = torch.clamp(conflicts, min=0)
                                    
                                    first_term_Lagrangian = objective*torch.matmul(s, conflicts.T)
                                    second_term_Lagrangian = torch.matmul(lambda_dual.view(K), (s*conflicts).T-delt)#)
                                    logarithm_probs = s_distribution.log_prob(s)
                                    log_probs_sum = torch.sum(logarithm_probs)

                                    Lagrangian = -((first_term_Lagrangian + second_term_Lagrangian - baseline)*log_probs_sum) # negative because of adam optim
                                    
                                    Lagrangian.backward()
                                    
                                    # update GNN parameters and zero the gradients
                                    if optimizer is not None:
                                        optimizer.step()
                                        optimizer.zero_grad()
                                    with torch.no_grad():
                                        baseline = baseline_alpha * baseline + (1 - baseline_alpha) * (first_term_Lagrangian+second_term_Lagrangian).item()
                                       
                                L.append(-Lagrangian.detach().cpu().numpy())  
                                L_first_term.append(first_term_Lagrangian.detach().cpu().numpy())
                                L_second_term.append(second_term_Lagrangian.detach().cpu().numpy())
                        k = k + 1

                    # save results from training                
                    results[phase, 'L'].append(np.mean(np.array(L)))
                    
                elif phase == 'test':
                    # testing
                    print("test")
                    model.eval()
                    of = []
                    mc = []
                    rc = []
                    average_violation = []
                    median_violation = []
                    min_violation = []
                    max_violation = []
                    res_value_min = []
                    res_value_max = []
                    res_value_mean = []
                    lambda_min = []
                    lambda_max = []
                    lambda_mean = []
                    lambda_mean_T_2 = []
                    tx_sum = []
                    succ_tx_sum = []
                    j = 0
                    for i, data in enumerate(loader['test']):
                        model.zero_grad()
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
                            # sample GNN and update lambda
                            for t in range(T):
                                node_probs = model(lambda_dual_gnn_test.detach(), scale, edge_index).view(-1, K)
                                s_distribution = torch.distributions.Bernoulli(node_probs)
                                s = s_distribution.sample().view(K)
                                conflicts = ones_vector - torch.matmul(A, s)
                                conflicts = torch.clamp(conflicts, min=0)
                                
                                total_tx = total_tx + s
                                succ_tx = succ_tx + s*conflicts

                                all_decisions.append(s)
                                
                                lambda_dual_test = lambda_dual_test - lr_lambda_use*((s*conflicts-delt).T
                                                                    +lambda_dual_test*weight_decay)
                                lambda_dual_test = torch.clamp(lambda_dual_test, min=0)
                                
                                lambda_dual_gnn_test = lambda_dual_test.repeat(batch_size, 1)  
                                lambda_dual_gnn_test = torch.clamp(lambda_dual_gnn_test, min=0).to(device) 

                                if weight_decay != 0 and t >= 0.95*T: # I average the last 5% lambdas
                                    lambdas_resilience.append(lambda_dual_test)
                                obj_function.append(torch.matmul(s, conflicts.T))
                                minimum_tx_constraint.append((s*conflicts).view(K))
                                
                                if epoch%lambda_granularity == 0 or epoch==num_epochs-1:
                                    dual_variable.append(lambda_dual_test)
                                if t == T/2:
                                    lambda_mean_T_2.append(torch.mean(lambda_dual_test.detach().cpu()).numpy())
                            
                            tx_sum.append(torch.sum(total_tx).cpu())
                            succ_tx_sum.append(torch.sum(succ_tx).cpu())

                            minimum_tx_constraint = torch.stack(minimum_tx_constraint).view(T, -1, K)
                            minimum_tx_constraint = (torch.mean(minimum_tx_constraint, dim=0)).view(K) # average over time
                            obj_function = torch.stack(obj_function)
                            obj_function = torch.mean(obj_function, dim=0) # average over time
                            of.append(obj_function.detach().cpu().numpy()*100/K)
                            violations = delt - minimum_tx_constraint.detach().cpu().numpy()
                            violation_mask = minimum_tx_constraint.detach().cpu().numpy() < delt
                            violations_violating = violations[violation_mask]
                            if len(violations_violating) > 0:
                                average_violation.append(np.mean(violations_violating))
                                median_violation.append(np.median(violations_violating))
                                max_violation.append(np.quantile(violations_violating, q=0.95))
                                min_violation.append(np.quantile(violations_violating, q=0.05))
                            else:
                                average_violation.append(0)
                                median_violation.append(0)
                                max_violation.append(0)
                                min_violation.append(0)

                            comparisons = (violation_mask).sum()
                            mc.append(comparisons*100/K)
                            if weight_decay != 0: # I average the last 5% lambdas:
                                lambdas_resilience = torch.stack(lambdas_resilience).view(int(T*0.05), -1, K)
                                lambdas_resilience = (torch.mean(lambdas_resilience, dim=0)).view(K) # average over time
                                resilient_constraint = (minimum_tx_constraint.detach().cpu().numpy() - delt + weight_decay*lambdas_resilience.detach().cpu().numpy() < 0).sum()
                                rc.append(resilient_constraint*100/K)
                                res_value_mean.append(torch.mean(weight_decay*lambdas_resilience.detach().cpu()).numpy())
                                res_value_min.append(torch.quantile(weight_decay*lambdas_resilience.detach().cpu(), q=0.05).numpy())
                                res_value_max.append(torch.quantile(weight_decay*lambdas_resilience.detach().cpu(), q=0.95).numpy())
                                
                            if epoch%lambda_granularity == 0 or epoch==num_epochs-1:
                                dual_variable = torch.stack(dual_variable) # I want to keep all T values for all edges.
                            lambda_mean.append(torch.mean(lambda_dual_test.detach().cpu()).numpy())
                            lambda_min.append(torch.quantile(lambda_dual_test.detach().cpu(), q=0.05).numpy())
                            lambda_max.append(torch.quantile(lambda_dual_test.detach().cpu(), q=0.95).numpy())
                        j = j + 1 # next graph

                    results[phase, 'minimum_tx_constraint'].append(np.mean(np.array(mc)))
                    results[phase, 'objective_function'].append(np.mean(np.array(of)))
                    results[phase, 'transmissions'].append(np.mean(np.array(tx_sum)))
                    results[phase, 'succ_transmissions'].append(np.mean(np.array(succ_tx_sum)))

                    if epoch%5 == 0:
                        of_train = []
                        mc_train = []
                        average_violation_train = []
                        median_violation_train = []
                        min_violation_train = []
                        max_violation_train = []
                        tx_sum_train = []
                        succ_tx_sum_train = []
                        rc_train = []
                        distributions = []
                        for x in range(num_graphs_train):
                            distributions.append([])
                        k = 0
                        for i, data in enumerate(loader['train']):
                            # compute dual variables for graphs in the training set to feed the model in the next epoch
                            if k < how_many_graphs:
                                model.zero_grad()
                                data.to(device)
                                
                                # get relevant fields from data
                                edge_index, K, A, Delta = get_data(data, batch_size)
                                #print(K)
                                ones_vector = torch.ones(1, K, dtype=torch.float32).to(device)
                                # print(K)
                                total_tx_train = torch.zeros(K).to(device)
                                succ_tx_train = torch.zeros(K).to(device)

                                with torch.set_grad_enabled(False):
                                    obj_function_train = []
                                    minimum_tx_constraint_train = []
                                    dual_variable_train = []
                                    lambdas_resilience_train = []

                                    lambda_dual_test = torch.zeros(K, 1).to(device) #
                                    lambda_dual_gnn_test = lambda_dual_test.repeat(batch_size, 1).to(device) # the GNN requires this shape,
                                                                                    # so we repeat for each batch
                                    # sample GNN and update lambda
                                    for t in range(T):
                                        node_probs = model(lambda_dual_gnn_test.detach(), scale, edge_index).view(-1, K)
                                        s_distribution = torch.distributions.Bernoulli(node_probs)
                                        s = s_distribution.sample().view(K)
                                        
                                        conflicts = ones_vector - torch.matmul(A, s)
                                        conflicts = torch.clamp(conflicts, min=0)
                                        
                                        total_tx_train = total_tx_train + s
                                        succ_tx_train = succ_tx_train + s*conflicts

                                        if (t+1) == 0:
                                            lambda_dual_test = lambda_dual_test - lr_lambda_use*((s*conflicts-delt).T
                                                                                +lambda_dual_test*weight_decay)
                                            lambda_dual_test = torch.clamp(lambda_dual_test, min=0)
                                            
                                            lambda_dual_gnn_test = lambda_dual_test.repeat(batch_size, 1)  
                                            lambda_dual_gnn_test = torch.clamp(lambda_dual_gnn_test, min=0).to(device) 
                                        if weight_decay != 0 and t >= 0.75*T: # I average the last 25% lambdas
                                            lambdas_resilience_train.append(lambda_dual_test)
                                        obj_function_train.append(torch.matmul(s, conflicts.T))
                                        minimum_tx_constraint_train.append((s*conflicts).view(K))
                                        distributions[i].append((lambda_dual_test.detach().cpu().numpy()))
                                    
                                    tx_sum_train.append(torch.sum(total_tx).cpu())
                                    succ_tx_sum_train.append(torch.sum(succ_tx).cpu())

                                    minimum_tx_constraint_train = torch.stack(minimum_tx_constraint_train).view(T, -1, K)
                                    minimum_tx_constraint_train = (torch.mean(minimum_tx_constraint_train, dim=0)).view(K) # average over time
                                    obj_function_train = torch.stack(obj_function_train)
                                    obj_function_train = torch.mean(obj_function_train, dim=0) # average over time
                                    of_train.append(obj_function_train.detach().cpu().numpy()*100/K)
                                    violations_train = delt - minimum_tx_constraint_train.detach().cpu().numpy()
                                    violation_mask_train = minimum_tx_constraint_train.detach().cpu().numpy() < delt
                                    violations_violating_train = violations_train[violation_mask_train]
                                    if len(violations_violating_train) > 0:
                                        average_violation_train.append(np.mean(violations_violating_train))
                                        max_violation_train.append(np.quantile(violations_violating_train, q=0.95))
                                        min_violation_train.append(np.quantile(violations_violating_train, q=0.05))
                                        median_violation_train.append(np.median(violations_violating_train))
                                    else:
                                        average_violation_train.append(0)
                                        max_violation_train.append(0)
                                        min_violation_train.append(0)
                                        median_violation_train.append(0)
                                    if weight_decay != 0: # I average the last 25% lambdas:
                                        lambdas_resilience_train = torch.stack(lambdas_resilience_train).view(int(T*0.25), -1, K)
                                        lambdas_resilience_train = (torch.mean(lambdas_resilience_train, dim=0)).view(K) # average over time
                                        resilient_constraint_train = (minimum_tx_constraint_train.detach().cpu().numpy() - delt + weight_decay*lambdas_resilience_train.detach().cpu().numpy() < 0).sum()
                                        rc_train.append(resilient_constraint_train*100/K)
                                    comparisons_train = (violation_mask_train).sum()
                                    mc_train.append(comparisons_train*100/K)
                                    
                            k = k + 1    

                    if weight_decay != 0:
                        results[phase, 'constraint_with_resilience'].append(np.mean(np.array(rc)))
                    
                    if epoch%lambda_granularity == 0 or epoch==num_epochs-1:
                        results[phase, 'lambda'].append(dual_variable.detach().cpu().numpy())
                        plot_lambdas(epoch, delta, lr_phi, lr_lambda, dual_variable, exp_name, weight_decay)
            
            if epoch%25 == 0 and epoch > 0 and updates < 3:
                baseline_alpha = baseline_alpha + 0.05
                
                updates = updates + 1
            
            torch.save(results, './results/{}.json'.format(exp_name))
            torch.save(model.state_dict(), './models/{}.pt'.format(exp_name))
    print('experiment finished')     

if __name__ == '__main__':
    def parse_float_list(list_str):
        """
        Expects a string like "[0.25, 0.5, 0.75]" and returns [0.25, 0.5, 0.75].
        """
        try:
            # Safely evaluate the string into a Python object
            value = ast.literal_eval(list_str)
            if not isinstance(value, (list, tuple)):
                raise ValueError
            # Convert elements to float
            return [float(x) for x in value]
        except (SyntaxError, ValueError):
            raise argparse.ArgumentTypeError(
                f"Invalid list of floats: '{list_str}'. "
                "Use a bracketed list like '[0.25, 0.5, 0.75]'."
            )
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
    parser.add_argument('--exps', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr_lambda', type=float, default=2)
    parser.add_argument('--lr_phi', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--delta', type=float, default=0.01)
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--how_many_graphs', type=int, default=10)
    parser.add_argument('--small', type=int, default=0)
    parser.add_argument('--degree', type=int, default=6)
    parser.add_argument('--norm', type=int, default=1)
    parser.add_argument('--num_features', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--norm_layer', type=str, default='layer')
    parser.add_argument('--hops', type=int, default=2)
    parser.add_argument('--gnn_orig', type=int, default=0)
    parser.add_argument('--plot', type=int, default=1)
    parser.add_argument('--obj_function', type=int, default=1)
    parser.add_argument('--add', type=int, default=0)
    parser.add_argument('--distributions', type=parse_float_list, 
                    default=[0.3, 0.6, 0.9])
    
    args = parser.parse_args()
    print(f'epochs: {args.epochs}')
    print(f'batch_size: {args.batch_size}')
    print(f'lr_lambda: {args.lr_lambda}')
    print(f'lr_phi: {args.lr_phi}')
    print(f'delta: {args.delta}')
    
    sawl(args, num_exps=args.exps, num_epochs=args.epochs, batch_size=args.batch_size, lr_lambda=args.lr_lambda, 
         lr_phi=args.lr_phi, weight_decay=args.weight_decay,  delta=args.delta, deg_lim=args.degree, 
         how_many_graphs=args.how_many_graphs, scale=args.scale, num_features=args.num_features, objective=args.obj_function,
         batch_norm=args.norm, gnn_original = args.gnn_orig, num_layers = args.num_layers, hops=args.hops, norm_layer=args.norm_layer,
         plot=args.plot, dist_lim=args.distributions, small_graph=args.small, added=args.add)
    

