import os
import sys

import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch.utils.data import DataLoader
import tqdm

from classes.cube_classes import Cube3
from generate.generate_states import char_to_move_index
import pytorch_classes.graph_dataset as gd
import pytorch_classes.color_dataset as cd
from pytorch_classes.config import CONFIGS

ACTION_DICT = {
    'qt': np.array(["U'", "U", "D'", "D", "L'", "L", "R'", "R", "F'", "F", "B'", "B"]),
    'ft': np.array(["U'", "U", "D'", "D", "L'", "L", "R'", "R", "F'", "F", "B'", "B", "U2", "D2", "L2", "R2", "F2", "B2"])
}
CURRFOLDER = '.'


def get_best_actions(generator):
    best_actions = []
    for gen in generator:
        last = gen[-1]
        if last[-1] != '2':
            last = last[:-1] if last[-1] == "'" else last + "'"
        best_actions.append(last)
    return best_actions


def change_cubestate(cube_state, operation, cube=None):
    if cube is None:
        cube = Cube3()
    if operation[-1] == '2':
        prev_state_ls = cube.prev_state([cube_state], char_to_move_index(operation[:-1]))[0]
        prev_state_ls = cube.prev_state([prev_state_ls], char_to_move_index(operation[:-1]))[0]
    else:
        prev_state_ls = cube.prev_state([cube_state], char_to_move_index(operation))[0]
    return prev_state_ls


def generate_cubestate(cube_generator):
    cube = Cube3()
    curr_state = cube.generate_goal_states(1)[0]
    for operation in cube_generator:
        curr_state = change_cubestate(curr_state, operation, cube=cube)
    return curr_state


def eval_color_single(colors, network):
    torch_input = torch.from_numpy(colors).double()
    return network(torch_input).flatten().item()


def eval_graph_single(colors, network):
    graph_data = gd.create_complete_graph_data_obj(colors, 0)
    return network(graph_data).flatten().item()


def is_prediction_correct(cube_state, network, metric, pred_dict, dist_dict, eval_single_func):
    preds, dists = [], []
    orig_colors = cube_state.colors // 9
    orig_immutable = tuple(orig_colors)
    orig_dist = dist_dict[orig_immutable]
    for action in ACTION_DICT[metric]:
        curr_state = change_cubestate(cube_state, action)
        input_colors = curr_state.colors // 9
        input_immutable = tuple(input_colors)
        if input_immutable in pred_dict:
            pred_distance = pred_dict[input_immutable]
            true_distance = dist_dict[input_immutable]
        else:
            pred_distance = eval_single_func(input_colors, network)
            true_distance = orig_dist + 1
            pred_dict[input_immutable] = pred_distance
            dist_dict[input_immutable] = true_distance
        preds.append(pred_distance)
        dists.append(true_distance)

    preds = np.array(preds)
    dists = np.array(dists)

    best_pred_dist = np.min(preds)
    pred_winners = np.argwhere(preds == best_pred_dist).flatten()
    best_true_dist = np.min(dists)
    dist_winners = np.argwhere(dists == best_true_dist).flatten()
    intersection = np.in1d(pred_winners, dist_winners, assume_unique=True)
    return np.sum(intersection) >= 1


def calc_accuracy(states, network, network_name, metric, pred_dict, dist_dict, eval_single_func):
    nr_of_states = len(states)
    print(f"Network {network_name}")
    correct = 0
    for state in tqdm.tqdm(states, miniters=1000):
        if is_prediction_correct(state, network, metric, pred_dict, dist_dict, eval_single_func):
            correct += 1
    acc = correct/nr_of_states
    print(f"Acc: {acc}, nr of examples {nr_of_states}, correct {correct}")
    return acc, correct, nr_of_states


def create_color_loader(dataframe, solved_state_colors):
    inputs = dataframe['colors'].tolist() + [solved_state_colors]
    targets = dataframe['distance'].tolist() + [0]
    dataset = cd.ColorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader


def create_graph_loader(dataframe, solved_state_colors):
    inputs = dataframe['colors'].tolist() + [solved_state_colors]
    targets = dataframe['distance'].tolist() + [0]
    dataset = gd.create_data_list(inputs, targets)
    dataloader = torch_geometric.loader.DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader


def eval_dataset(network, dataloader, colors_ls):
    predicted_dict, distance_dict = {}, {}
    network.eval()
    network.double()
    for colors, data in tqdm.tqdm(zip(colors_ls, dataloader), miniters=1000):
        if len(data) == 2:
            inputs, distance = data
        else:
            inputs, distance = data, data.y.squeeze()
        outputs = network(inputs).flatten().item()
        color_immutable = tuple(colors)
        predicted_dict[color_immutable] = outputs
        distance_dict[color_immutable] = distance
    return predicted_dict, distance_dict


def load_model(path):
    checkpoint = torch.load(path)
    ModelClass = checkpoint['model_class']
    hyperparams = checkpoint['model_hyperparams']
    model = ModelClass(**hyperparams)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def single_accuracy_n_moves(states, colors, dataloader, metric, dataset_name, model_name, test_size, random_seed,
                            eval_single_func):
    path = f'{CURRFOLDER}/checkpoints/{dataset_name}/{model_name}/model_rs{random_seed}_ts{test_size:.1f}.pth'
    network = load_model(path)
    pred_dict, dist_dict = eval_dataset(network, dataloader, colors)
    acc, correct, nr_of_states = calc_accuracy(states, network, f'{model_name}, t.s. {test_size}, r.s. {random_seed}',
                                               metric, pred_dict, dist_dict, eval_single_func)
    return acc, correct, nr_of_states


def collect_accuracies_n_moves(states, colors, dataloader, metric, dataset_name, model_name, eval_single_func):
    nr_of_states = len(states)
    print(f"Testing {nr_of_states} of examples.")
    result_dict = {}
    for test_size in np.arange(0.1, 1.0, 0.1):
        result_dict[test_size] = []
        for random_seed in range(10):
            acc, _, _ = single_accuracy_n_moves(states, colors, dataloader, metric, dataset_name, model_name,
                                                                 test_size, random_seed, eval_single_func)
            result_dict[test_size].append(acc)
    return result_dict


def run_resnet_n_moves(n=5):
    df = pd.read_pickle(f'{CURRFOLDER}/data/processed/{n}_moves_dataset_single.pkl')
    # df = df[df['distance'] <= 1]
    solved_colors = (Cube3().generate_goal_states(1)[0].colors // 9).tolist()
    loader = create_color_loader(df, solved_colors)
    resnet_acc_dict = collect_accuracies_n_moves(df.state, df.colors.tolist() + [solved_colors], loader,
                                                 'qt', '5moves', 'ResNet', eval_color_single)
    print(resnet_acc_dict)


def run_symnet_n_moves(n=5):
    df = pd.read_pickle(f'{CURRFOLDER}/data/processed/{n}_moves_dataset_single.pkl')
    # df = df[df['distance'] <= 1]
    solved_colors = (Cube3().generate_goal_states(1)[0].colors // 9).tolist()
    loader = create_graph_loader(df, solved_colors)
    resnet_acc_dict = collect_accuracies_n_moves(df.state, df.colors.tolist() + [solved_colors], loader,
                                                 'qt', f'{n}moves', 'SymEqNet', eval_graph_single)
    print(resnet_acc_dict)


if __name__ == '__main__':
    # run_resnet_n_moves()
    # run_symnet_n_moves()

    MODEL = 'SymEqNet'

    if MODEL == 'ResNet':
        eval_func = eval_color_single
        create_loader = create_color_loader
    else:
        eval_func = eval_graph_single
        create_loader = create_graph_loader

    CONFIG_NR = int(sys.argv[1])
    rnd_seed, tst_size = CONFIGS[CONFIG_NR]

    df = pd.read_pickle(f'{CURRFOLDER}/data/processed/5_moves_dataset_single.pkl')
    solved_colors = (Cube3().generate_goal_states(1)[0].colors // 9).tolist()
    loader = create_loader(df, solved_colors)
    acc, correct, nr_of_states = single_accuracy_n_moves(df.state, df.colors.tolist() + [solved_colors], loader,
                                                         'qt', '5moves', MODEL, tst_size, rnd_seed, eval_func)
    print(f'Accuracy: {acc}')
    os.makedirs(f'{CURRFOLDER}/data/evals/{MODEL}/', exist_ok=True)
    with open(f'{CURRFOLDER}/data/evals/{MODEL}/ts{tst_size}_rs{rnd_seed}.txt', 'w') as f:
        f.write('\n'.join([str(acc), str(correct), str(nr_of_states)]))

# TODO: kociemba
