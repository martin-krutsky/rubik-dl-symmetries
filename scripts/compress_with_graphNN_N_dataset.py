import ast
from collections import defaultdict
import functools
import math
import numpy as np
import pandas as pd
import pickle
import random
from tqdm import tqdm

from scipy.spatial import distance_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.graphgym import global_mean_pool

from classes.cube_classes import Cube3State, Cube3
from generate.generate_states import ids_to_color
from utils.random_seed import seed_worker, seed_all, init_weights
from utils.compressions import *


torch.set_default_dtype(torch.float64)


print('script started')
N_MOVES = 6
df = pd.read_csv(f'data/{N_MOVES}_moves_dataset_single.csv', index_col=0, converters={'colors':ast.literal_eval, 'generator':ast.literal_eval})

def indices_to_position(indices):
    pos_array = np.array([
        [0.5, 0.5, 0], [1.5, 0.5, 0], [2.5, 0.5, 0],
        [0.5, 1.5, 0], [1.5, 1.5, 0], [2.5, 1.5, 0],
        [0.5, 2.5, 0], [1.5, 2.5, 0], [2.5, 2.5, 0],
        
        [2.5, 0.5, 3], [1.5, 0.5, 3], [0.5, 0.5, 3],
        [2.5, 1.5, 3], [1.5, 1.5, 3], [0.5, 1.5, 3],
        [2.5, 2.5, 3], [1.5, 2.5, 3], [0.5, 2.5, 3],
        
        [2.5, 0, 2.5], [2.5, 0, 1.5], [2.5, 0, 0.5],
        [1.5, 0, 2.5], [1.5, 0, 1.5], [1.5, 0, 0.5],
        [0.5, 0, 2.5], [0.5, 0, 1.5], [0.5, 0, 0.5],
        
        [0.5, 3, 2.5], [0.5, 3, 1.5], [0.5, 3, 0.5],
        [1.5, 3, 2.5], [1.5, 3, 1.5], [1.5, 3, 0.5],
        [2.5, 3, 2.5], [2.5, 3, 1.5], [2.5, 3, 0.5],
        
        [3, 2.5, 2.5], [3, 2.5, 1.5], [3, 2.5, 0.5],
        [3, 1.5, 2.5], [3, 1.5, 1.5], [3, 1.5, 0.5],
        [3, 0.5, 2.5], [3, 0.5, 1.5], [3, 0.5, 0.5],
        
        [0, 0.5, 2.5], [0, 0.5, 1.5], [0, 0.5, 0.5],
        [0, 1.5, 2.5], [0, 1.5, 1.5], [0, 1.5, 0.5],
        [0, 2.5, 2.5], [0, 2.5, 1.5], [0, 2.5, 0.5],
    ])
    return pos_array[np.array(indices)]
    

class GCNNet(torch.nn.Module):
    def __init__(self, hidden_channels, node_features_size=9):
        super(GCNNet, self).__init__()
        self.conv1 = gnn.SimpleConv()
#         self.conv1 = gnn.GCNConv(node_features_size, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)

        x = global_mean_pool(x, data.batch if data.batch is not None else torch.zeros((data.x.size(0)), dtype=torch.int64))
        x = self.lin(x)
        return x

# ----------------------------------------------------------------

@functools.lru_cache
def calc_distances(filtered_indices):
    vertices = indices_to_position(filtered_indices)
    dist_mat = distance_matrix(vertices, vertices)
    dist_mat = np.rint(dist_mat*10e4).astype(int)
    return dist_mat

    
def create_complete_graph_data_obj(colors, distance_from_goal, node_features_size=9, verbose=False, aggregate=False, for_hashing=False):
    indices = np.arange(54)
    colors = np.array(colors)
    node_features = np.zeros((54, node_features_size), dtype=int)
    edge_index = []
    edge_attr = []
    curr_idx = 0
    for color in range(6):
        filtered_indices = indices[colors == color]
        dist_mat = calc_distances(tuple(filtered_indices))
        node_features[filtered_indices] = dist_mat
        for i, node_i in enumerate(filtered_indices):
            for j, node_j in enumerate(filtered_indices):
#                 if i == j:
#                     continue
                edge_index.append([node_i, node_j])
                edge_attr.append(dist_mat[i,j])
        curr_idx += len(filtered_indices)
    y = distance_from_goal
    node_features = np.sort(node_features, axis=1)
    data = Data(x=torch.tensor(node_features, dtype=int), edge_index=torch.tensor(edge_index).T.long() , edge_attr=torch.tensor(edge_attr, dtype=int).unsqueeze(1), y=torch.Tensor([y]))
    return data


networksCompGraph = create_networks(NetworkClass=GCNNet, network_args={'hidden_channels': 9}, num_of_networks=10)
gcn_compressions = calculate_all_dicts_from_activations(df=df, max_distance=N_MOVES, input_handling_func=create_complete_graph_data_obj, networks=networksCompGraph, is_graph_nn=True)


with open(f'data/temp/gcn_compressions_distance{N_MOVES}.pkl', 'wb') as f:
    pickle.dump(gcn_compressions, f)
    
set_intersections_activations_sparse_graph = compute_set_intersections(gcn_compressions)

plot_distance_compressions(gcn_compressions, f'gcn_compressions/from_activations{N_MOVES}')
