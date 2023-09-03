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
    def __init__(self, hidden_channels, node_features_size=10):
        super(GCNNet, self).__init__()
        self.conv1 = gnn.GCNConv(node_features_size, hidden_channels)
        self.conv2 = gnn.GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)
        x = global_mean_pool(x, data.batch if data.batch is not None else torch.zeros((data.x.size(0)), dtype=torch.int64))
        x = self.lin(x)
        return x

# ----------------------------------------------------------------

@functools.lru_cache
def calc_distances(filtered_indices):
    vertices = indices_to_position(filtered_indices)
    dist_mat = distance_matrix(vertices, vertices)  # pouze se stredem
    return dist_mat

def find_middle(vertices):
    for vertex in vertices:
        if (vertex[0] == vertex[1] == 1.5) or (vertex[0] == vertex[2] == 1.5) or (vertex[1] == vertex[2] == 1.5):
            return vertex
    return None
    
def create_sparse_graph_data_obj(colors, distance_from_goal, node_features_size, verbose=False, aggregate=False, for_hashing=False):
    indices = np.arange(54)
    colors = np.array(colors)
    node_features = np.ones((54, node_features_size))
    edge_index = []
    edge_attr = []
    curr_idx = 0
    for color in range(6):
        filtered_indices = indices[colors == color]
        position_filtered_indices = indices_to_position(filtered_indices)
        middle = find_middle(position_filtered_indices)
        distances = distance_matrix(position_filtered_indices, middle.reshape(1, -1))
        ctr = 0
        for i in range(len(filtered_indices)):
            if (position_filtered_indices[i] == middle).all():
                continue
            node_i = curr_idx + ctr + 1
            node_j = curr_idx
            edge_index.append([node_i, node_j])
            edge_attr.append(distances[i].item())
            edge_index.append([node_j, node_i])
            edge_attr.append(distances[i].item())
            ctr += 1
        curr_idx += len(filtered_indices)
    y = distance_from_goal
    data = Data(x=torch.Tensor(node_features), edge_index=torch.Tensor(edge_index).T.long() , edge_attr=torch.Tensor(edge_attr).unsqueeze(1), y=torch.Tensor([y]))
    return data


# data_list_sparse_graph = []
# for _, row in tqdm(df.iterrows()):
#     data_list_sparse_graph.append(create_sparse_graph_data_obj(row['colors'], row['distance'], node_features_size=10))
# 
# loader_sparseGraph = DataLoader(data_list_sparse_graph, batch_size=32)
# 
# netSparseGraph = GCNNet(10)
# 
# ## Training loop
# optimizer = torch.optim.Adam(netSparseGraph.parameters(), lr=0.001)
# 
# for epoch in range(10):
#     netSparseGraph.train()
#     total = 0
#     total_loss = 0.0
#     for i, data in tqdm(enumerate(loader_sparseGraph)):
#         optimizer.zero_grad()
#         out = netSparseGraph(data)
#         loss = F.mse_loss(out.squeeze(), data.y.squeeze(), reduction='sum')
#         loss.backward()
#         total_loss += loss.item()
#         total += data.y.size(0)
#         optimizer.step()
#     if (epoch+1) % 1 == 0:
#         print(f'Epoch {epoch+1}: train loss {total_loss/total:0.4f}')

networksSparseGraph = create_networks(NetworkClass=GCNNet, network_args={'hidden_channels': 16}, num_of_networks=10)
distance_all_acts_sparse_graph = calculate_all_dicts_from_activations(df=df, max_distance=N_MOVES, input_handling_func=create_sparse_graph_data_obj, networks=networksSparseGraph, is_graph_nn=True)


with open(f'data/temp/distance{N_MOVES}_all_acts_sparse_graph.pkl', 'wb') as f:
    pickle.dump(distance_all_acts_sparse_graph, f)
    
set_intersections_activations_sparse_graph = compute_set_intersections(distance_all_acts_sparse_graph)

plot_distance_compressions(distance_all_acts_sparse_graph, f'sparseGraphNet/from_activations{N_MOVES}')

# ----------------------------------------------------------------
print('===========================================================')

@functools.lru_cache
def calc_distances(filtered_indices):
    vertices = indices_to_position(filtered_indices)
    dist_mat = distance_matrix(vertices, vertices)  # pouze se stredem
    return dist_mat

def create_complete_graph_data_obj(colors, distance_from_goal, node_features_size, verbose=False, aggregate=False, for_hashing=False):
    indices = np.arange(54)
    colors = np.array(colors)
    node_features = np.ones((54, node_features_size))
    edge_index = []
    edge_attr = []
    curr_idx = 0
    for color in range(6):
        filtered_indices = indices[colors == color]
        dist_mat = calc_distances(tuple(filtered_indices))
        for i in range(len(filtered_indices)):
            for j in range(len(filtered_indices)):
                node_i = curr_idx + i
                node_j = curr_idx + j
                edge_index.append([node_i, node_j])
                edge_attr.append(dist_mat[i,j])
        curr_idx += len(filtered_indices)
    y = distance_from_goal
    data = Data(x=torch.Tensor(node_features), edge_index=torch.Tensor(edge_index).T.long() , edge_attr=torch.Tensor(edge_attr).unsqueeze(1), y=torch.Tensor([y]))
    return data


# data_list_complete_graph = []
# for _, row in tqdm(df.iterrows()):
#     data_list_complete_graph.append(create_complete_graph_data_obj(row['colors'], row['distance'], node_features_size=10))
# 
# loader_compGraph = DataLoader(data_list_complete_graph, batch_size=32)
# 
# netCompGraph = GCNNet(10)
# 
# ## Training loop
# optimizer = torch.optim.Adam(netCompGraph.parameters(), lr=0.001)
# 
# for epoch in range(10):
#     netCompGraph.train()
#     total = 0
#     total_loss = 0.0
#     for i, data in tqdm(enumerate(loader_compGraph)):
#         optimizer.zero_grad()
#         out = netCompGraph(data)
#         loss = F.mse_loss(out.squeeze(), data.y.squeeze(), reduction='sum')
#         loss.backward()
#         total_loss += loss.item()
#         total += data.y.size(0)
#         optimizer.step()
#     if (epoch+1) % 1 == 0:
#         print(f'Epoch {epoch+1}: train loss {total_loss/total:0.4f}')

networksCompGraph = create_networks(NetworkClass=GCNNet, network_args={'hidden_channels': 16}, num_of_networks=10)
distance_all_acts_complete_graph = calculate_all_dicts_from_activations(df=df, max_distance=N_MOVES, input_handling_func=create_complete_graph_data_obj, networks=networksCompGraph, is_graph_nn=True)

with open(f'data/temp/distance{N_MOVES}_all_acts_complete_graph.pkl', 'wb') as f:
    pickle.dump(distance_all_acts_complete_graph, f)
   
set_intersections_activations_complete_graph = compute_set_intersections(distance_all_acts_complete_graph)

plot_distance_compressions(distance_all_acts_complete_graph, f'completeGraphNet/from_activations{N_MOVES}')

# ----------------------------------------------------------------
