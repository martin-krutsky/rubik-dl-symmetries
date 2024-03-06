import ast
import functools
import pickle
import sys

from scipy.spatial import ConvexHull, distance_matrix

from utils.compressions import *
from generate.symmetry_config import pos_list


torch.set_default_dtype(torch.float64)


print('script started')
df = pd.read_csv(f'data/processed/kociemba_dataset.csv', index_col=0, converters={'colors':ast.literal_eval, 'generator':ast.literal_eval})


def indices_to_position(indices):
    pos_array = np.array(pos_list)
    return pos_array[np.array(indices)]

# ----------------------------------------------------------------

@functools.lru_cache
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


@functools.lru_cache
def calc_distances_pairwise_sum(vertices):
    vertices = indices_to_position(vertices)
    distances = distance_matrix(vertices, vertices)
    distances = np.sort(distances.sum(axis=1))
    distances = np.rint(distances * 10e4).astype(int)
    return distances


def calc_distances_pairwise_lex(vertices):
    vertices = indices_to_position(vertices)
    distances = distance_matrix(vertices, vertices)
    print(distances)
    distances = np.sort(distances, axis=1)
    print(distances)
    distance_indices = np.lexsort(np.rot90(distances))
    print(distance_indices)
    distances = distances[distance_indices].flatten()
    print(distances)
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


# ----------------------------------------------------------------
# ----------------------------------------------------------------

counterVolume = 0


def calc_volumes(colors, verbose=True, aggregate=False, for_hashing=False):
    if verbose:
        global counterVolume
        counterVolume += 1
        if (counterVolume + 1) % 1000 == 0:
            print(counterVolume + 1)
    volumes = []
    indices = np.arange(54)
    colors = np.array(colors)
    for color in range(6):
        filtered_indices = indices[colors == color]
        volume = calc_volume(tuple(filtered_indices))
        volumes.append(volume)
    volumes = np.array(volumes)
    if aggregate:
        volumes = np.sum(volumes, dtype=np.double)[..., np.newaxis]
    return volumes


counterDist = 0


def calc_all_distances(colors, distance_func=calc_distances_middle, verbose=True, aggregate=False, for_hashing=False):
    if verbose:
        global counterDist
        counterDist += 1
        if (counterDist + 1) % 1000 == 0:
            print(counterDist + 1)
    distances_ls = []
    indices = np.arange(54)
    colors = np.array(colors)
    for color in range(6):
        filtered_indices = indices[colors == color]
        distances = distance_func(tuple(filtered_indices))
        distances_ls.append(distances)
    distances_ls = np.array(distances_ls)
    if aggregate:
        distances_ls = np.sum(distances_ls, axis=0, dtype=np.double)
    return distances_ls

# ----------------------------------------------------------------


if __name__ == "__main__":
    HEURISTIC = int(sys.argv[1])
    heuristic_dict = {
        0: 'vol', 1: 'd1', 2: 'd2', 3: 'd3', 4: 'd4', 5: 'd5'
    }
    heuristic_function = None
    if HEURISTIC == 0:
        heuristic_function = calc_volumes
    elif HEURISTIC == 1:
        heuristic_function = functools.partial(calc_all_distances, distance_func=calc_distances_middle)
    elif HEURISTIC == 2:
        heuristic_function = functools.partial(calc_all_distances, distance_func=calc_distances_pairwise_sum)
    elif HEURISTIC == 3:
        heuristic_function = functools.partial(calc_all_distances, distance_func=calc_distances_pairwise_lex)
    elif HEURISTIC == 4:
        heuristic_function = functools.partial(calc_all_distances, distance_func=calc_distances_manh_pairwise_sum)
    elif HEURISTIC == 5:
        heuristic_function = functools.partial(calc_all_distances, distance_func=calc_distances_manh_pairwise_lex)

    compressions, hash_to_sizes = calculate_all_dicts_from_data(df=df, max_distance=None,
                                                                input_handling_func=heuristic_function)

    with open(f'data/temp/{heuristic_dict[HEURISTIC]}_compressions_kociemba.pkl', 'wb') as f:
        pickle.dump(compressions, f)

    with open(f'data/temp/{heuristic_dict[HEURISTIC]}_hashtosizes_kociemba.pkl', 'wb') as f:
        pickle.dump(hash_to_sizes, f)

    # set_intersections_activations_sparse_graph = compute_set_intersections(gcn_compressions)
    # plot_distance_compressions(gcn_compressions, f'symeqnet_compressions/from_activations{N_MOVES}')
