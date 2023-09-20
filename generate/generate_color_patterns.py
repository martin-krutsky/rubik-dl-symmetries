from itertools import combinations
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from .symmetry_config import *

middle_indices = [4 + i * 9 for i in range(6)]
edge_indices = [i for i in range(54) if (((i % 9) % 2) == 0) and ((i % 9) != 4)]
corner_indices = [i for i in range(54) if (((i % 9) % 2) == 1)]


def rotate_indices(indices, direction):
    return np.vectorize(rotation_dict[direction].get)(indices)


def all_rotations_of_indices(indices):
    rotated = [frozenset(indices)]
    for rotation in rotation_symmetry_generators:
        new_indices = indices.copy()
        for direction in rotation:
            new_indices = rotate_indices(new_indices, direction)
        rotated.append(frozenset(new_indices))  
    return rotated


def reflect_indices(indices):
    return np.vectorize(reflection_dict.get)(indices)


def generate_all_symmetric(indices):
    all_rotated = all_rotations_of_indices(indices)
    reflected = reflect_indices(indices)
    all_rotated_reflected = all_rotations_of_indices(reflected)
    all_symmetric = set(all_rotated) | set(all_rotated_reflected)
    return all_symmetric


def save_as_df(colors, symmetry_classes, start_index, seq_counter, dataset_split, folder):
    df = pd.DataFrame({
        'index': list(range(start_index, start_index+len(colors))),
        'colors': colors.tolist(),
        'symmetry_class': symmetry_classes
    })
    df.to_csv(os.path.join(folder, f'color_patterns/color_pattern_dataset{seq_counter}.csv'))
    print(f'Saved file {seq_counter + 1}/{dataset_split*6}.')


def assign_symmetry_classes(folder, filename, middle_idx_nr):
    middle_index = middle_indices[middle_idx_nr]
    dataset = np.load(os.path.join(folder, filename))
    dataset_size = len(dataset)
    dataset_split = 100
    
    hashset = set()
    colors, symmetry_classes, sym_counter, start_index, seq_counter = [], [], 0, 0, 0
    processed_counter = 0
    
    for i, color_list in enumerate(tqdm(dataset)):
        if (i + 1) % 10000 == 0:
            print('Processed', processed_counter)
        frozen_color_list = frozenset(color_list)
        frozen_color_hash = hash(frozen_color_list)
        if frozen_color_hash not in hashset:
            all_symmetric = generate_all_symmetric(color_list)
            for j, sym in enumerate(all_symmetric):
                sym_hash = hash(sym)
                if middle_index in sym:
                    hashset.add(sym_hash)
                colors.append(np.array(list(sym), dtype=np.uint8))
                symmetry_classes.append(sym_counter)
                processed_counter += 1
            sym_counter += 1
            
            if len(symmetry_classes) >= (dataset_size/dataset_split):
                save_as_df(np.array(colors), symmetry_classes, start_index, seq_counter, dataset_split, folder)
                start_index += len(symmetry_classes)
                seq_counter += 1
                colors, symmetry_classes = [], []
        else:
            hashset.remove(frozen_color_hash)
    save_as_df(np.array(colors), symmetry_classes, start_index, seq_counter, dataset_split, folder)


def generate_color_indices(folder, filename, middle_idx_nr):
    middle_index = middle_indices[middle_idx_nr]
    edge_combs = list(combinations(edge_indices, 4))
    corner_combs = list(combinations(corner_indices, 4))

    print(len(edge_combs), len(corner_combs))

    dataset = []
    for edge_subset in tqdm(edge_combs):
        inner_dataset = []
        for corner_subset in corner_combs:
            color_list = [middle_index] + list(edge_subset) + list(corner_subset)
            color_list = sorted(color_list)
            inner_dataset.append(color_list)
        dataset.append(np.array(inner_dataset, dtype=np.uint8))
    dataset = np.concatenate(dataset, axis=0)

    np.save(os.path.join(folder, filename), dataset)
    

def visualize_color_indices(indices):
    indices = np.array(indices)
    cross_copy = np.array(positions_on_cross_list[0])
    for index in indices:
        cross_copy[cross_copy == index] = -2
    cross_copy[(cross_copy != -1) & (cross_copy != -2)] = 0
    cross_list = cross_copy.flatten().tolist()
    map_dict = {-1: ' ', -2: '\033[91mX', 0: '\033[90m-'}
    str_arr = np.array([map_dict[item] for item in cross_list]).reshape(cross_copy.shape)
    str_vis = '\n'.join([''.join(row) for row in str_arr])
    return str_vis
    