from itertools import combinations
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from tqdm import tqdm

from generate.symmetry_config import *
from generate.generate_color_patterns import *

from utils.compressions import compress_for_color


def indices_to_position(indices):
    pos_array = np.array(pos_list)
    return pos_array[indices]


def calc_distances_weisfeiler_lehman(vertices):
    vertices = indices_to_position(vertices)
    distances = distance_matrix(vertices, vertices)
    for message_pass in range(1):
        new_distances = distances.copy()
        for i in range(len(vertices)):
            new_distances += distances[:, i].reshape(1, -1) * distances[i].reshape(-1, 1)
        distances = new_distances
    distances = np.sort(distances.sum(axis=1))
    distances = np.rint(distances*10e4).astype(int)
    return distances


def compress_one_dataset(filepath, compression_dict):
    df = pd.read_csv(filepath)
    df.colors = df.colors.map(eval)
    df.head()
    
    compression_dict = compress_for_color(df, calc_distances_weisfeiler_lehman, break_on_error=False, compression_dict=compression_dict)
    return compression_dict


DIR = 'data/color_patterns/'

compression_dict = None
for root, dirs, filenames in os.walk(DIR):
    for i, filename in enumerate(sorted(filenames)):
        print(f'Processing dataset from file {filename}')  # nr. {i+1}/{len(filenames)}
        filepath = os.path.join(root, filename)
        compression_dict = compress_one_dataset(filepath, compression_dict)