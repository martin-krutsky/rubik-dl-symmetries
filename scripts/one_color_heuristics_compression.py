import functools
import pickle
import sys
from scipy.spatial import ConvexHull, distance_matrix

from generate.generate_color_patterns import *
from utils.compressions import compress_for_color


def indices_to_position(indices):
    pos_array = np.array(pos_list)
    return pos_array[indices]


def calc_volume(vertices):
    vertices = indices_to_position(vertices)
    if (vertices[:,0] == vertices[0,0]).all() or (vertices[:,1] == vertices[0,1]).all() or (vertices[:,2] == vertices[0,2]).all():
        volume = 0
    else:
        volume = ConvexHull(vertices).volume
    volume = np.rint(volume*10e4).astype(int)
    return volume


def find_middle(vertices):
    for vertex in vertices:
        if (vertex[0] == vertex[1] == 1.5) or (vertex[0] == vertex[2] == 1.5) or (vertex[1] == vertex[2] == 1.5):
            return vertex
    return None

def calc_distances_middle(vertices):
    vertices = indices_to_position(vertices)
    middle = find_middle(vertices)
    distances = np.linalg.norm(vertices - middle.reshape(1, -1), axis=1)
    distances = distances[distances != 0]
    distances = np.sort(distances)
    distances = np.rint(distances*10e4).astype(int)
    return distances


def calc_distances_pairwise_sum(vertices):
    vertices = indices_to_position(vertices)
    distances = distance_matrix(vertices, vertices)
    distances = np.sort(distances.sum(axis=1))
    distances = np.rint(distances * 10e4).astype(int)
    return distances


def calc_distances_pairwise_lex(vertices):
    vertices = indices_to_position(vertices)
    distances = distance_matrix(vertices, vertices)
    distances = np.sort(distances, axis=1)
    distance_indices = np.lexsort(np.rot90(distances))
    distances = distances[distance_indices].flatten()
    distances = np.rint(distances*10e4).astype(int)
    return distances


def calc_distances_manh_pairwise_sum(vertices):
    vertices = indices_to_position(vertices)
    distances = distance_matrix(vertices, vertices, p=1)
    distances = np.sort(distances.sum(axis=1))
    distances = np.rint(distances * 10e4).astype(int)
    return distances


def calc_distances_manh_pairwise_lex(vertices):
    vertices = indices_to_position(vertices)
    distances = distance_matrix(vertices, vertices, p=1)
    distances = np.sort(distances, axis=1)
    distance_indices = np.lexsort(np.rot90(distances))
    distances = distances[distance_indices].flatten()
    distances = np.rint(distances*10e4).astype(int)
    return distances


prohibited_together = [()]

def is_impossible(colors):
    


def compress_one_dataset(filepath, compression_dict, hash_to_sizes, heuristics_func):
    df = pd.read_csv(filepath)
    df.colors = df.colors.map(eval)
    df.head()

    compression_dict = compress_for_color(df, heuristics_func, 
                                          break_on_error=False, verbose=False,
                                          compression_dict=compression_dict,
                                          hash_to_sizes=hash_to_sizes)
    return compression_dict


DIR = 'data/color_patterns/'
HEURISTIC = int(sys.argv[1])

heuristic_function = None
if HEURISTIC == 0:
    heuristic_function = calc_volume
elif HEURISTIC == 1:
    heuristic_function = calc_distances_middle
elif HEURISTIC == 2:
    heuristic_function = calc_distances_pairwise_sum
elif HEURISTIC == 3:
    heuristic_function = calc_distances_pairwise_lex
elif HEURISTIC == 4:
    heuristic_function = calc_distances_manh_pairwise_sum
elif HEURISTIC == 5:
    heuristic_function = calc_distances_manh_pairwise_lex

compression_dictionary, hash_to_sizes = None, None
total_patterns = 0
classes_ls = []
for root, dirs, filenames in os.walk(DIR):
    for i, filename in enumerate(sorted(filenames)):
        print(f'Processing dataset from file {filename}')  # nr. {i+1}/{len(filenames)}
        file_path = os.path.join(root, filename)
        
        df = pd.read_csv(file_path)
        total_patterns += len(df.index)
        classes_ls += list(df['symmetry_class'].unique())
        df.colors = df.colors.map(eval)
        compression_dictionary, hash_to_sizes = compress_one_dataset(file_path, compression_dictionary, hash_to_sizes, heuristic_function)

print(f'Total number of patterns: {total_patterns}')
print(f'Total number of classes: {len(set(classes_ls))}')

with open(f'data/processed/hash_to_sizes_{HEURISTIC}.pickle', 'wb') as f:
    pickle.dump(hash_to_sizes, f, protocol=pickle.HIGHEST_PROTOCOL)

nonunique_num = 0
incorrectly_interchangeable = 0

for value_list in hash_to_sizes.values():
    if len(value_list) > 1:
        nonunique_num += len(value_list)
        value_list_sum = sum(value_list)
        for value in value_list:
            incorrectly_interchangeable += value * (value_list_sum - value)
            
print(f'Nr of classes with non-unique hashes: {nonunique_num}')
print(f'Nr of incorrectly interchangeable members: {incorrectly_interchangeable}')
