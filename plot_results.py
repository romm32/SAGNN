import torch
from matplotlib import pyplot as plt
import os
import random
import argparse
import numpy as np
import os

def plot_1(versions, T):
    plt.rcParams.update({'font.size': 20})
    T_0 = 1
    print('generating data')
    print("----------------------------")
    deg_lim = 6
    colors = {'0.1': 'navy', '0.125': 'darkred', '0.15': 'darkgreen'}

    plt.figure(figsize=(10, 8))
    for delt in [0.1]:  # , 0.125, 0.15]:
        obj_function = [None] 
        min_tx_constraint = [None] 

        if delt == 0.1:
            exp_name = ('exp_0_epochs_100_delta_0.1_lr_phi_5e-05_lr_lambda_2_'
                        'weight_decay0.05_degree_6_of_1')

        results = torch.load('results/' + exp_name + '.json')
        obj_function = results[('test', 'objective_function')]
        min_tx_constraint = results[('test', 'minimum_tx_constraint')]

        epochs = len(obj_function)
        x = np.arange(epochs)
        obj_function = np.array(obj_function).squeeze()
        plt.plot(x, obj_function, label='SAGNN, $\Delta={}$'.format(delt),
                 color=colors[str(delt)], linewidth=2.0)

    plt.xlabel('Training epochs')
    plt.ylabel('Percentage of successful transmissions')
    plt.legend(loc='best', ncol=1, fontsize='16')
    plt.ylim(0, 26)
    plt.xlim(0, 100)
    image_path_svg = os.path.join('plots/', f'plot1-objective{deg_lim}.svg')
    image_path_png = os.path.join('plots/', f'plot1-objective{deg_lim}.png')
    plt.savefig(image_path_svg, format='svg', bbox_inches='tight', transparent=True)
    plt.savefig(image_path_png, dpi=600, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 8))
    obj_function = [None] 
    min_tx_constraint = [None] 
    if delt == 0.1:
        exp_name = ('exp_0_epochs_100_delta_0.1_lr_phi_5e-05_lr_lambda_2_'
                    'weight_decay0.05_degree_6_of_1')

    results = torch.load('results/' + exp_name + '.json')
    obj_function = results[('test', 'objective_function')]
    min_tx_constraint = results[('test', 'minimum_tx_constraint')]

    min_tx_constraint = np.array(min_tx_constraint).squeeze()
    x = np.arange(len(min_tx_constraint))
    

    plt.plot(x, min_tx_constraint, label='SAGNN, $\Delta={}$'.format(delt),
                color=colors[str(delt)])
    plt.xlabel('Training epochs')
    plt.ylabel('Percentage of links violating constraint')
    plt.legend(loc='upper right')
    plt.ylim(0, 60)
    plt.xlim(0, 100)
    image_path = os.path.join('plots/', 'plot1-constraints' + str(deg_lim) + '.svg')
    image_path_png = os.path.join('plots/', 'plot1-constraints' + str(deg_lim) + '.png')
    plt.savefig(image_path,
            format='svg', bbox_inches='tight', transparent=True)
    plt.savefig(image_path_png,
                dpi=600, bbox_inches='tight')  
    plt.close()


if __name__ == '__main__':

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
    parser.add_argument('--plot', type=list, default='')
    parser.add_argument('--versions', type=str, default='')
    
    args = parser.parse_args()
    print(f'plots: {args.plot}')
    print(f'versions: {args.versions}')

    T = 200
    for num in args.plot:
        if num == '1':
            plot_1(args.versions, T)
