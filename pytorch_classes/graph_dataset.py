import functools
import numpy as np
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from scipy.spatial import distance_matrix
from tqdm import tqdm

from .config import POS_ARRAY


def indices_to_position(indices):
    return POS_ARRAY[np.array(indices)]


@functools.lru_cache
def calc_distances(filtered_indices):
    vertices = indices_to_position(filtered_indices)
    dist_mat = distance_matrix(vertices, vertices)
    # dist_mat = np.rint(dist_mat*10e4).astype(int)
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
                edge_index.append([node_i, node_j])
                edge_attr.append(dist_mat[i,j])
        curr_idx += len(filtered_indices)
    y = distance_from_goal
    node_features = np.sort(node_features, axis=1)
    data = Data(x=torch.tensor(node_features).double(), edge_index=torch.tensor(edge_index).T.long(),
                edge_attr=torch.tensor(edge_attr).double().unsqueeze(1), y=torch.tensor([y]))
    return data


def create_data_list(colors_list, target_list):
    data_list_complete_graph = []
    for colors, target in tqdm(zip(colors_list, target_list)):
        data_list_complete_graph.append(create_complete_graph_data_obj(colors, target))
    return data_list_complete_graph
