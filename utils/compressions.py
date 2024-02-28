from collections import defaultdict
from typing import List, Callable, DefaultDict, Dict, Set, Tuple, Type, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from utils.random_seed import seed_all


def create_networks(
        network_class: Type[torch.nn.Module], network_args=None, num_of_networks: int = 10
) -> List[torch.nn.Module]:
    """
    Initialize N neural networks of given class, each with different seed.
    """
    if network_args is None:
        network_args = {}
    networks = []
    for i in range(num_of_networks):
        seed_all(i)
        curr_net = network_class(**network_args)
        networks.append(curr_net)
    return networks


# --- COMPRESSION FOR SINGLE COLOR DATASET ---


def compress_for_color(
        df: pd.DataFrame, input_handling_func: Callable, break_on_error: bool = True, verbose: bool = True,
        compression_dict: Dict[Tuple, Tuple] = None, hash_to_sizes = None
) -> Tuple[Dict[Tuple, Tuple], Dict[Tuple, int]]:
    """
    Create a compression dictionary for a dataset of facelets corresponding to a single color in the Rubik's cube.
    Alert on possible intersections/non-unique hashes.
    """
    if compression_dict is None:
        compression_dict = dict()
        hash_to_sizes = defaultdict(list)
    for sym_class in tqdm(df.symmetry_class.unique()):
        df_filtered = df[df['symmetry_class'] == sym_class]
        indices = df_filtered['index']
        colors = df_filtered['colors']
        sym_class_comp_set = set()
        processed_data = None
        for ind, color_indices in zip(indices, colors):
            processed_data = input_handling_func(color_indices)
            if isinstance(processed_data, np.ndarray):
                processed_data = hash(processed_data.tostring())
            sym_class_comp_set.add(processed_data)
        if len(sym_class_comp_set) > 1:
            if verbose:
                print('ERROR: Multiple hashes in a symmetry class', sym_class)
            if break_on_error:
                break

        if processed_data in compression_dict:
            if verbose:
                print(f'ERROR: Same hashes ({processed_data}) for symmetry class', sym_class, 'and',
                      compression_dict[processed_data])
            if break_on_error:
                break
        compression_dict[processed_data] = sym_class
        hash_to_sizes[processed_data].append(len(indices))
    return compression_dict, hash_to_sizes


# --- COMPRESSION DICTIONARIES ---


def create_dicts_from_data(df: pd.DataFrame, distance: int, input_handling_func: Callable) -> DefaultDict[Tuple, List]:
    """
    Use dataset of cube states and their generators to produce a dictionary of compression classes
    based on the processed data (e.g., volumes, distances, angles, etc.).
    """
    distance_x_dict = defaultdict(list)
    hash_to_sizes = defaultdict(list)
    df_x = df[df['distance'] == distance]
    for class_id in sorted(df_x['class_id'].unique()):
        distance_x = df_x[df_x['class_id'] == class_id]['colors']
        distance_x_gens = df_x[df_x['class_id'] == class_id]['generator']
        distance_x_cls = df_x[df_x['class_id'] == class_id]['class_id']
    
        curr_proc_data_dict = defaultdict(int)
        for i, (cube, cube_gen, cube_cls) in enumerate(tqdm(list(zip(distance_x, distance_x_gens, distance_x_cls)))):
            processed_data = input_handling_func(cube, verbose=False, aggregate=True, for_hashing=True)
            if isinstance(processed_data[0], np.ndarray):
                processed_data = [hash(sub_array.tostring()) for sub_array in processed_data]
            processed_data = sorted(processed_data)
            distance_x_dict[tuple(processed_data)].append((i, cube_gen, cube_cls))
            curr_proc_data_dict[tuple(processed_data)] += 1
            
        for proc_dat in curr_proc_data_dict:
            hash_to_sizes[proc_dat].append(curr_proc_data_dict[proc_dat])
    return distance_x_dict, hash_to_sizes


def calculate_all_dicts_from_data(
        df: pd.DataFrame, max_distance: int, input_handling_func: Callable
) -> List[DefaultDict[Tuple, List]]:
    """
    Compute compression dictionaries for cubes with distance from goal in range between 1 and max_distance.
    """
    distance_all_dicts = []
    hash_to_cls_all_dicts = []
    if max_distance is None:
        for distance in sorted(df['distance'].unique()):
            distance_x_dict, hash_to_cls_x_dict = create_dicts_from_data(df, distance, input_handling_func)
            distance_all_dicts.append(distance_x_dict)
            hash_to_cls_all_dicts.append(hash_to_cls_x_dict)            
    else:
        for distance in range(1, max_distance + 1):
            distance_x_dict, hash_to_cls_x_dict = create_dicts_from_data(df, distance, input_handling_func)
            distance_all_dicts.append(distance_x_dict)
            hash_to_cls_all_dicts.append(hash_to_cls_x_dict)
    return distance_all_dicts, hash_to_cls_all_dicts


def create_dicts_from_activations(
        df: pd.DataFrame, distance: int, input_handling_func: Callable,
        networks: List[torch.nn.Module], is_graph_nn: bool = False, node_features_size=9
) -> DefaultDict[Tuple, List]:
    """
    Use dataset of cube states and their generators to produce a dictionary of compression classes
    based on forward activations of randomly initialized untrained neural networks.
    """
    distance_x = df[df['distance'] == distance]['colors']
    distance_x_gens = df[df['distance'] == distance]['generator']
    distance_x_dist = df[df['distance'] == distance]['distance']
    distance_x_dict = defaultdict(list)
    for i, (cube, cube_gen, cube_dist) in enumerate(tqdm(list(zip(distance_x, distance_x_gens, distance_x_dist)))):
        activations = []
        for network in networks:
            network.eval()
            if is_graph_nn:
                prepared_data = input_handling_func(cube, cube_dist, node_features_size=node_features_size,
                                                    verbose=False, aggregate=False, for_hashing=False)
                activation = float(np.squeeze(network(prepared_data).detach().numpy()))
            else:
                activation = float(np.squeeze(network(torch.tensor(
                    input_handling_func(cube, verbose=False, aggregate=False, for_hashing=False))).detach().numpy()))

            activation = int(activation * 1e8)
            activations.append(activation)
        distance_x_dict[tuple(activations)].append((i, cube_gen))
    return distance_x_dict


def calculate_all_dicts_from_activations(
        df: pd.DataFrame, max_distance: Optional[int], input_handling_func: Callable,
        networks: List[torch.nn.Module], is_graph_nn: bool = False
) -> List[DefaultDict[Tuple, List]]:
    """
    Compute compression dictionaries for cubes with distance from goal in range between 1 and max_distance.
    """
    if max_distance is None:
        distance_all_dicts = [
            create_dicts_from_activations(df, distance, input_handling_func, networks, is_graph_nn=is_graph_nn) for
            distance in sorted(df['distance'].unique())]
    else:
        distance_all_dicts = [
            create_dicts_from_activations(df, distance, input_handling_func, networks, is_graph_nn=is_graph_nn) for
            distance in range(1, max_distance + 1)]
    return distance_all_dicts


# --- SET INTERSECTION ---


def compute_set_intersections(
        distance_all_acts: List[DefaultDict[Tuple, List]], distances: Optional[List[int]] = None
) -> Dict[Tuple, Set]:
    """
    Given compression dictionaries, calculate their possible intersection, i.e. the NN's inability
    to distinguish between compression classes with cubes with different distance to goal.
    """
    intersections = {}
    for i in range(len(distance_all_acts)):
        for j in range(i + 1, len(distance_all_acts)):
            if distances is None:
                intersections[(i + 1, j + 1)] = set(distance_all_acts[i].keys()) & set(distance_all_acts[j].keys())
                print(f'Intersection size between sets {i + 1} AND {j + 1}: {len(intersections[(i + 1, j + 1)])}')
            else:
                intersections[(distances[i], distances[j])] = set(distance_all_acts[i].keys()) & set(
                    distance_all_acts[j].keys())
                print(f'Intersection size between sets {distances[i]} AND {distances[j]}: '
                      f'{len(intersections[(distances[i], distances[j])])}')
    return intersections


# --- PLOTTING ---


def plot_histo(data: List[int], filename: str, visible_bins: int = 20):
    """
    Plot a histogram of cube distributions.
    """
    plt.figure(figsize=(7, 4))
    nr_of_bins = max(data)
    sns.histplot(data, bins=[i for i in range(50)])
    x_ticks = [i for i in range(0, 52, 2)]
    plt.xticks(x_ticks)
    plt.xlim((0,49))
    plt.xlabel("Compression size",fontsize=12)
    plt.ylabel("Count",fontsize=12)

    plt.savefig(f'{filename}.png', dpi=300)
    plt.savefig(f'{filename}.pdf', format='pdf')
    plt.show()
    plt.close()


def plot_distance_compressions(distance_all_acts: List[DefaultDict[Any, List]], model_name: str,
                               distances: Optional[List[int]] = None):
    """
    Plot histograms of compression sizes 1. separately for each distance, 2. for all cubes.
    """
    all_compressions = []
    iterable = distances if distances is not None else range(len(distance_all_acts))
    for i in iterable:
        dist_acts = distance_all_acts[i]
        act_class_compressions = list(map(len, dist_acts.values()))
        all_compressions += act_class_compressions
        plot_histo(act_class_compressions, f'imgs/{model_name}/{i + 1}moves_activ_class_sizes_histo.png')
    plot_histo(all_compressions, f'imgs/{model_name}/all_moves_activ_class_sizes_histo.png')


def plot_class_ids_per_compressions(
        distance_all_acts: List[DefaultDict[float, List]], df: pd.DataFrame, model_name: str
):
    """
    Plot histograms of ground-truth class ids per each compression class in distance_all_acts
    """
    all_classes_counts = []
    for i in range(len(distance_all_acts)):
        dist_acts = distance_all_acts[i]
        classes_counts = []
        for key, values in dist_acts.items():
            classes = []
            for _, generator in values:
                for index, row in df.iterrows():
                    if generator[0] in row['generator']:
                        classes.append(row['class_id'])
                        break
            classes_counts.append(len(set(classes)))
        all_classes_counts += classes_counts
        plot_histo(classes_counts, f'imgs/{model_name}/{i + 1}moves_class_counts_per_activs.png')
    plot_histo(all_classes_counts, f'imgs/{model_name}/all_moves_class_counts_per_activs.png')
