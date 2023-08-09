from collections import defaultdict
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from classes.cube_classes import Cube3State, Cube3
from generate.generate_states import ids_to_color
from utils.random_seed import seed_worker, seed_all, init_weights
from utils.compressions import *

import functools
from scipy.spatial import ConvexHull

torch.set_default_dtype(torch.float64)


N_MOVES = 6
df = pd.read_pickle(f'data/{N_MOVES}_moves_dataset_single.pkl')

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

# ----------------------------------------------------------------

@functools.lru_cache
def calc_volume(filtered_indices):
    vertices = indices_to_position(filtered_indices)
    if (vertices[:,0] == vertices[0,0]).all() or (vertices[:,1] == vertices[0,1]).all() or (vertices[:,2] == vertices[0,2]).all():
        volume = 0
    else:
        volume = ConvexHull(vertices).volume
    return volume

counterVolume = 0

def calc_volumes(colors, verbose=True, aggregate=False, for_hashing=False):
    if verbose:
        global counterVolume
        counterVolume += 1
        if (counterVolume + 1) % 1000 == 0:
            print(counterVolume+1)
    volumes = []
    indices = np.arange(54)
    colors = np.array(colors)
    for color in range(6):
        filtered_indices = indices[colors == color]
        volume = calc_volume(tuple(filtered_indices))
        if for_hashing:
            volume = np.rint(volume*10e4).astype(int)
        volumes.append(volume)
    volumes = np.array(volumes)
    if aggregate:
        volumes = np.sum(volumes, dtype=np.double)[..., np.newaxis]
    return volumes


distance_all_data_volume = calculate_all_dicts_from_data(df=df, max_distance=N_MOVES, input_handling_func=calc_volumes)

with open(f'data/temp/distance{N_MOVES}_all_data_volume_single.pkl', 'wb') as f:
    pickle.dump(distance_all_data_volume, f)
    
set_intersections_data_volume = compute_set_intersections(distance_all_data_volume)

plot_distance_compressions(distance_all_data_volume, f'volumeNet/from_data{N_MOVES}')

# ----------------------------------------------------------------

def find_middle(vertices):
    for vertex in vertices:
        if (vertex[0] == vertex[1] == 1.5) or (vertex[0] == vertex[2] == 1.5) or (vertex[1] == vertex[2] == 1.5):
            return vertex
    return None

@functools.lru_cache
def calc_distances(filtered_indices):
    vertices = indices_to_position(filtered_indices)
    middle = find_middle(vertices)
    distances = np.linalg.norm(vertices - middle.reshape(1, -1), axis=1)
    distances = distances[distances != 0]
    # zprojektovat a pak secist?
    distances = np.sort(distances)
    return distances

counterDist = 0

def calc_all_distances(colors, verbose=True, aggregate=False, for_hashing=False):
    if verbose:
        global counterDist
        counterDist += 1
        if (counterDist + 1) % 1000 == 0:
            print(counterDist+1)
    distances_ls = []
    indices = np.arange(54)
    colors = np.array(colors)
    for color in range(6):
        filtered_indices = indices[colors == color]
        distances = calc_distances(tuple(filtered_indices))
        if for_hashing:
            distances = np.rint(distances*10e4).astype(int)
        distances_ls.append(distances)
    distances_ls = np.array(distances_ls)
    if aggregate:
        distances_ls = np.sum(distances_ls, axis=0, dtype=np.double)
    return distances_ls


distance_all_data_dist = calculate_all_dicts_from_data(df=df, max_distance=N_MOVES, input_handling_func=calc_all_distances)

with open(f'data/temp/distance{N_MOVES}_all_data_dist_single.pkl', 'wb') as f:
    pickle.dump(distance_all_data_dist, f)
    
set_intersections_data_dist = compute_set_intersections(distance_all_data_dist)

plot_distance_compressions(distance_all_data_dist, f'distNet/from_data{N_MOVES}')

# ----------------------------------------------------------------

def unit_vector(vector):
    """ Returns the unit vector of the vector. """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2' """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.abs(np.arccos(np.dot(v1_u, v2_u)))

@functools.lru_cache
def calc_angles(filtered_indices):
    vertices = indices_to_position(filtered_indices)
    middle = find_middle(vertices)
    distances = np.linalg.norm(vertices - middle.reshape(1, -1), axis=1)
    vertices = vertices[distances != 0]  # - middle,  np.zeros(3)
    angles = [angle_between(v, middle) for v in vertices]
    angles = np.sort(angles)
    return angles

counterAngles = 0

def calc_all_angles(colors, verbose=True, aggregate=False, for_hashing=False):
    if verbose:
        global counterAngles
        counterAngles += 1
        if (counterAngles + 1) % 10000 == 0:
            print(counterAngles+1)
    angles_ls = []
    indices = np.arange(54)
    colors = np.array(colors)
    for color in range(6):
#         print('color', color)
        filtered_indices = indices[colors == color]
        angles = calc_angles(tuple(filtered_indices))
        if for_hashing:
            angles = np.rint(angles*10e4).astype(int)
#         print(angles)
        angles_ls.append(angles)
#         print('---')
    angles_ls = np.array(angles_ls)
    if aggregate:
        angles_ls = np.sum(angles_ls, axis=0, dtype=np.double)
#     print('=================')
    return angles_ls


distance_all_data_angle = calculate_all_dicts_from_data(df=df, max_distance=N_MOVES, input_handling_func=calc_all_angles)

with open(f'data/temp/distance{N_MOVES}_all_data_angle_single.pkl', 'wb') as f:
    pickle.dump(distance_all_data_angle, f)
    
set_intersections_data_angle = compute_set_intersections(distance_all_data_angle)

plot_distance_compressions(distance_all_data_angle, 'angleNet/from_data')
