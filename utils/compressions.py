from collections import defaultdict
from typing import List, Callable, Dict, Set, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from utils.random_seed import seed_worker, seed_all


def create_networks(NetworkClass: Type[torch.nn.Module], network_args: dict = {}, num_of_networks: int = 10) -> List[torch.nn.Module]:
    '''
    Initialize N neural networks of given class, each with different seed.
    '''
    networks = []
    for i in range(num_of_networks):
        seed_all(i)
        curr_net = NetworkClass(**network_args)
        networks.append(curr_net)
    return networks


# --- COMPRESSION DICTIONARIES ---


def create_dicts_from_data(df: pd.DataFrame, distance: int, input_handling_func: Callable) -> Dict[Tuple, Tuple]:
    '''
    Use dataset of cube states and their generators to produce a dictionary of compression classes
    based on the processed data (e.g., volumes, distances, angles, etc.).
    '''
    distance_x = df[df['distance'] == distance]['colors']
    distance_x_gens = df[df['distance'] == distance]['generator']
    distance_x_dict = defaultdict(list)
    for i, (cube, cube_gen) in enumerate(tqdm(list(zip(distance_x, distance_x_gens)))):
        processed_data = input_handling_func(cube, verbose=False)
        distance_x_dict[tuple(processed_data)].append((i, cube_gen))
    return distance_x_dict


def calculate_all_dicts_from_data(df: pd.DataFrame, max_distance: int, input_handling_func: Callable) -> List[Dict[Tuple, Tuple]]:
    '''
    Compute compression dictionaries for cubes with distance from goal in range between 1 and max_distance.
    '''
    distance_all_dicts = [create_dicts_from_data(df, distance, input_handling_func) for distance in range(1, max_distance + 1)]
    return distance_all_dicts


def create_dicts_from_activations(df: pd.DataFrame, distance: int, input_handling_func: Callable, networks: List[torch.nn.Module]) -> Dict[Tuple, Tuple]:
    '''
    Use dataset of cube states and their generators to produce a dictionary of compression classes
    based on forward activations of randomly initialized untrained neural networks.
    '''
    distance_x = df[df['distance'] == distance]['colors']
    distance_x_gens = df[df['distance'] == distance]['generator']
    distance_x_dict = defaultdict(list)
    for i, (cube, cube_gen) in enumerate(tqdm(list(zip(distance_x, distance_x_gens)))):
        activations = []
        for network in networks:
            activation = float(np.squeeze(network(torch.tensor(input_handling_func(cube, verbose=False))).detach().numpy()))
            activation = int(activation * 1e8)
            activations.append(activation)
        distance_x_dict[tuple(activations)].append((i, cube_gen))
    return distance_x_dict


def calculate_all_dicts_from_activations(df: pd.DataFrame, max_distance: int, input_handling_func: Callable, networks: List[torch.nn.Module]) -> List[Dict[Tuple, Tuple]]:
    '''
    Compute compression dictionaries for cubes with distance from goal in range between 1 and max_distance.
    '''
    distance_all_dicts = [create_dicts_from_activations(df, distance, input_handling_func, networks) for distance in range(1, max_distance + 1)]
    return distance_all_dicts


# --- SET INTERSECTION ---


def compute_set_intersections(distance_all_acts: List[Dict[Tuple, Tuple]]) -> Dict[Tuple, Set]:
    '''
    Given compression dictionaries, calculate their possible intersection, i.e. the NN's inability
    to distinguish between compression classes with cubes with different distance to goal.
    '''
    intersections = {}
    for i in range(len(distance_all_acts)):
        for j in range(i+1, len(distance_all_acts)):
            intersections[(i+1,j+1)] = set(distance_all_acts[i].keys()) & set(distance_all_acts[j].keys())
            print(f'Intersection size between sets {i+1} AND {j+1}: {len(intersections[(i+1,j+1)])}')
    return intersections


# --- PlOTTING ---


def plot_histo(data, filename, visible_bins=20):
    '''
    Plot a histogram of cube distributions.
    '''
    plt.figure(figsize=(10,5))
    nr_of_bins = max(data)
    plot = sns.histplot(data, bins=nr_of_bins)
    x_ticks = [i for i in range(2, nr_of_bins+1, max(nr_of_bins//visible_bins, 1))]
    plt.xticks(x_ticks)
    
    plt.savefig(filename)
    plt.show()


def plot_distance_compressions(distance_all_acts, model_name):
    '''
    Plot histograms of compression sizes 1. separately for each distance, 2. for all cubes.
    '''
    all_compressions = []
    for i in range(len(distance_all_acts)):
        dist_acts = distance_all_acts[i]
        act_class_compressions = list(map(len, dist_acts.values()))
        all_compressions += act_class_compressions
        plot_histo(act_class_compressions, f'imgs/{model_name}/{i+1}moves_activ_class_sizes_histo.png')
    plot_histo(all_compressions, f'imgs/{model_name}/all_moves_activ_class_sizes_histo.png')


def plot_class_ids_per_compressions(distance_all_acts, df, model_name):
    all_classes_counts = []
    for i in range(len(distance_all_acts)):
        dist_acts = distance_all_acts[i]
        classes_counts = []
        for key, values in dist_acts.items():
            classes = []
            for j, _ in values:
                classes.append(df.iloc[j]['class_id'])
            classes_counts.append(len(set(classes)))
        all_classes_counts += classes_counts
        plot_histo(classes_counts, f'imgs/{model_name}/{i+1}moves_class_counts_per_activs.png')
    plot_histo(all_classes_counts, f'imgs/{model_name}/all_moves_class_counts_per_activs.png')
