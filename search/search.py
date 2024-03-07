import ast
import functools
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch_geometric
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import tqdm

from classes.cube_classes import Cube3
from generate.generate_states import char_to_move_index
import pytorch_classes.graph_dataset as gd
import pytorch_classes.color_dataset as cd
from pytorch_classes.config import CONFIGS

ACTION_DICT = {
    'qt': np.array(["U'", "U", "D'", "D", "L'", "L", "R'", "R", "F'", "F", "B'", "B"]),
    'ft': np.array(
        ["U'", "U", "D'", "D", "L'", "L", "R'", "R", "F'", "F", "B'", "B", "U2", "D2", "L2", "R2", "F2", "B2"])
}


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


def eval_color_single(colors, network):
    torch_input = torch.from_numpy(colors).double()
    return network(torch_input).flatten().item()


def eval_graph_single(colors, network):
    graph_data = gd.create_complete_graph_data_obj(colors, 0)
    return network(graph_data).flatten().item()


def is_prediction_correct(cube_state, generator, network, metric, dataset_name, pred_dict, dist_dict, eval_single_func):
    preds, best_preds, dists = [], [], []
    orig_colors = cube_state.colors // 9
    orig_immutable = tuple(orig_colors)
    orig_dist = dist_dict[orig_immutable]
    best_actions = get_best_actions(generator)
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
        if action in best_actions:
            best_preds.append(pred_distance)

    preds = np.array(preds)
    dists = np.array(dists)

    if dataset_name == '5moves':
        best_pred_dist = np.min(preds)
        pred_winners = np.argwhere(preds == best_pred_dist).flatten()
        best_true_dist = np.min(dists)
        dist_winners = np.argwhere(dists == best_true_dist).flatten()
        intersection = np.in1d(pred_winners, dist_winners, assume_unique=True)
        return np.sum(intersection) >= 1, pred_dict, dist_dict
    elif 'kociemba' in dataset_name:
        return np.allclose(np.min(preds), np.min(best_preds)), pred_dict, dist_dict
    else:
        raise Exception


def calc_accuracy(states, generators, network, network_name, metric, dataset_name, pred_dict, dist_dict,
                  eval_single_func):
    n_states = len(states)
    print(f"Network {network_name}")
    n_correct = 0
    for state, generator in tqdm.tqdm(zip(states, generators), miniters=1000):
        is_correct, pred_dict, dist_dict = is_prediction_correct(
            state, generator, network, metric, dataset_name, pred_dict, dist_dict, eval_single_func
        )
        if is_correct:
            n_correct += 1
    accuracy = n_correct/n_states
    print(f"Acc: {accuracy}, nr of examples {n_states}, correct {n_correct}")
    return accuracy, n_correct, n_states


def get_train_test_set(dataframe, test_size, random_seed):
    train, test = train_test_split(
        dataframe, stratify=dataframe['distance'], test_size=test_size, random_state=random_seed
    )
    return train, test


def create_color_loader(dataframe, solved_state_colors, test_size, random_seed):
    train, test = get_train_test_set(dataframe, test_size, random_seed)

    test_inputs = test['colors'].tolist()
    test_targets = test['distance'].tolist()
    test_dataset = cd.ColorDataset(test_inputs, test_targets)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    train_inputs = train['colors'].tolist() + [solved_state_colors]
    train_targets = train['distance'].tolist() + [0]
    train_dataset = cd.ColorDataset(train_inputs, train_targets)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    return train_dataloader, test_dataloader, train, test


def create_graph_loader(dataframe, solved_state_colors, test_size, random_seed):
    train, test = get_train_test_set(dataframe, test_size, random_seed)

    test_inputs = test['colors'].tolist()
    test_targets = test['distance'].tolist()
    test_dataset = gd.create_data_list(test_inputs, test_targets)
    test_dataloader = torch_geometric.loader.DataLoader(test_dataset, batch_size=1, shuffle=False)

    train_inputs = train['colors'].tolist() + [solved_state_colors]
    train_targets = train['distance'].tolist() + [0]
    train_dataset = gd.create_data_list(train_inputs, train_targets)
    train_dataloader = torch_geometric.loader.DataLoader(train_dataset, batch_size=1, shuffle=False)
    return train_dataloader, test_dataloader, train, test


def eval_dataset(network, train_loader, test_loader, train_color_ls, test_color_ls):
    predicted_dict, distance_dict = {}, {}
    network.eval()
    network.double()
    for dataloader, colors_ls in zip([train_loader, test_loader], [train_color_ls, test_color_ls]):
        for colors, data in tqdm.tqdm(zip(colors_ls, dataloader), miniters=1000):
            if len(data) == 2:
                inputs, distance = data
            else:
                inputs, distance = data, data.y.squeeze()
            outputs = network(inputs).flatten().item()
            color_immutable = tuple(colors)
            predicted_dict[color_immutable] = outputs
            distance_dict[color_immutable] = distance.flatten().item()
    return predicted_dict, distance_dict


def load_model(path):
    checkpoint = torch.load(path)
    ModelClass = checkpoint['model_class']
    hyperparams = checkpoint['model_hyperparams']
    model = ModelClass(**hyperparams)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def single_accuracy_n_moves(states, generators, train_colors, test_colors, train_loader, test_loader, metric,
                            dataset_name, model_name, test_size, random_seed, eval_single_func):
    path = f'{CURRFOLDER}/checkpoints/{dataset_name}/{model_name}/model_rs{random_seed}_ts{test_size:.1f}.pth'
    network = load_model(path)
    pred_dict, dist_dict = eval_dataset(network, train_loader, test_loader, train_colors, test_colors)
    accuracy, n_correct, n_states = calc_accuracy(states, generators, network,
                                                  f'{model_name}, t.s. {test_size:.1f}, r.s. {random_seed}',
                                                  metric, dataset_name, pred_dict, dist_dict, eval_single_func)
    return accuracy, n_correct, n_states


if __name__ == '__main__':
    CURRFOLDER = '.'
    TASK = '5moves'
    MODEL = 'ResNet'

    if TASK == '5moves':
        FILE = '5_moves_dataset_single.pkl'
        METRIC = 'qt'
        pandas_reader = pd.read_pickle
    elif TASK == 'kociemba10':
        FILE = 'kociemba10_dataset.pkl'
        METRIC = 'ft'
        pandas_reader = pd.read_pickle
    elif TASK == 'kociemba':
        FILE = 'kociemba_dataset.csv'
        METRIC = 'ft'
        pandas_reader = functools.partial(pd.read_csv, index_col=0,
                                          converters={'colors': ast.literal_eval, 'distance': ast.literal_eval})
    else:
        raise Exception

    if MODEL == 'ResNet':
        eval_func = eval_color_single
        create_loader = create_color_loader
    else:
        eval_func = eval_graph_single
        create_loader = create_graph_loader

    CONFIG_NR = int(sys.argv[1])
    rnd_seed, tst_size = CONFIGS[CONFIG_NR]

    df = pandas_reader(f'{CURRFOLDER}/data/processed/{FILE}')
    # df = df[df['distance'] <= 1]
    solved_colors = (Cube3().generate_goal_states(1)[0].colors // 9).tolist()
    tr_loader, ts_loader, train_df, test_df = create_loader(df, solved_colors, tst_size, rnd_seed)
    acc, correct, nr_of_states = single_accuracy_n_moves(df.state, df.generator,
                                                         train_df.colors, test_df.colors.tolist() + [solved_colors],
                                                         tr_loader, ts_loader, METRIC, TASK, MODEL, tst_size, rnd_seed,
                                                         eval_func)
    print(f'Accuracy: {acc}')
    out_folder = f'{CURRFOLDER}/data/evals/{TASK}/{MODEL}'
    os.makedirs(out_folder, exist_ok=True)
    with open(f'{out_folder}/ts{tst_size:.1f}_rs{rnd_seed}.txt', 'w') as f:
        f.write('\n'.join([str(acc), str(correct), str(nr_of_states)]))
