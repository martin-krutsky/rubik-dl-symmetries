import copy

import numpy as np
import torch
from sklearn.model_selection import train_test_split
import tqdm

from classes.cube_classes import Cube3, Cube3State
from generate.generate_states import char_to_move_index
import pytorch_classes.graph_dataset as gd

ACTION_DICT = {
    'qt': np.array(["U'", "U", "D'", "D", "L'", "L", "R'", "R", "F'", "F", "B'", "B"]),
    'ft': np.array(
        ["U'", "U", "D'", "D", "L'", "L", "R'", "R", "F'", "F", "B'", "B", "U2", "D2", "L2", "R2", "F2", "B2"])
}
global_cube = Cube3()
SOLVED_STATE_COLORS = Cube3().generate_goal_states(1)[0].colors // 9


def is_goal_colors(colors):
    return np.array_equal(colors, SOLVED_STATE_COLORS)


def get_best_actions(generator):
    best_actions = []
    for gen in generator:
        last = gen[-1]
        if last[-1] != '2':
            last = last[:-1] if last[-1] == "'" else last + "'"
        best_actions.append(last)
    return best_actions


def create_cubestate_from_gen(generator):
    state = global_cube.generate_goal_states(1)[0]
    for move in generator[0]:
        state = change_cubestate(state, move)
    return state


def create_cubestate_from_colors(colors):
    return Cube3State(colors=colors)


def change_cubestate(cube_state, operation):
    new_cube_state = copy.copy(cube_state)
    if operation[-1] == '2':
        prev_state_ls = global_cube.prev_state([new_cube_state], char_to_move_index(operation[:-1]))[0]
        prev_state_ls = global_cube.prev_state([prev_state_ls], char_to_move_index(operation[:-1]))[0]
    else:
        prev_state_ls = global_cube.prev_state([new_cube_state], char_to_move_index(operation))[0]
    return prev_state_ls


def eval_color_single(colors, network):
    torch_input = torch.from_numpy(colors).double()
    return network(torch_input).flatten().item()


def eval_graph_single(colors, network):
    graph_data = gd.create_complete_graph_data_obj(colors, 0)
    return network(graph_data).flatten().item()


def search_bounded(cube_state, generator, network, metric, pred_dict, eval_single_func, bound=20):
    counter = 0
    goal_found = False
    path = []
    old_state = cube_state
    while not goal_found and counter < bound:
        preds = []
        new_states = []
        for action in ACTION_DICT[metric]:
            new_state = change_cubestate(old_state, action)
            new_states.append(new_state)
            input_colors = new_state.colors // 9
            if is_goal_colors(input_colors):
                path.append(action)
                return True, len(path) == len(generator[0]), path
            input_immutable = tuple(input_colors)
            if input_immutable in pred_dict:
                pred_distance = pred_dict[input_immutable]
            else:
                pred_distance = eval_single_func(input_colors, network)
                pred_dict[input_immutable] = pred_distance
            preds.append(pred_distance)
        old_state = new_states[np.argmin(preds)]
    return False, False, None


def calc_solved_rates(states, generators, network, network_name, metric, pred_dict, eval_single_func):
    n_states = len(states)
    print(f"Network {network_name}")
    solved, perfectly_solved = 0, 0
    for state, generator in tqdm.tqdm(zip(states, generators), miniters=1000):
        is_solved, is_perf_solved, path = search_bounded(state, generator, network, metric, pred_dict, eval_single_func)
        if is_solved:
            solved += 1
            if is_perf_solved:
                perfectly_solved += 1
    solved_rate = solved / n_states
    perf_solved_rate = perfectly_solved / n_states
    print(f"Solved rate: {solved_rate}, Perf. solved rate: {perf_solved_rate}, nr of examples {n_states}")
    return solved_rate, perf_solved_rate, solved, perfectly_solved, n_states


def next_if_correct(cube_state, generator, network, metric, pred_dict, dist_dict, eval_single_func):
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

    best_pred_dist = np.min(preds)
    pred_winners = np.argwhere(preds == best_pred_dist).flatten()
    best_true_dist = np.min(dists)
    dist_winners = np.argwhere(dists == best_true_dist).flatten()
    intersection = set(pred_winners) & set(dist_winners)
    if intersection:
        return ACTION_DICT[metric][next(iter(intersection))], pred_dict, dist_dict
    return False, pred_dict, dist_dict


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


def single_solved_rate_n_moves(states, generators, train_colors, test_colors, train_loader, test_loader, metric,
                               dataset_name, model_name, test_size, random_seed, eval_single_func, curr_folder):
    path = f'{curr_folder}/checkpoints/{dataset_name}/{model_name}/model_rs{random_seed}_ts{test_size:.1f}.pth'
    network = load_model(path)
    pred_dict, dist_dict = eval_dataset(network, train_loader, test_loader, train_colors, test_colors)
    solved_rate, perf_solved_rate, solved, perfectly_solved, n_states = calc_solved_rates(
        states, generators, network, f'{model_name}, t.s. {test_size:.1f}, r.s. {random_seed}',
        metric, pred_dict, eval_single_func
    )
    return solved_rate, perf_solved_rate, solved, perfectly_solved, n_states


def single_accuracy_n_moves(states, generators, train_colors, test_colors, train_loader, test_loader, metric,
                            dataset_name, model_name, test_size, random_seed, eval_single_func, curr_folder):
    path = f'{curr_folder}/checkpoints/{dataset_name}/{model_name}/model_rs{random_seed}_ts{test_size:.1f}.pth'
    network = load_model(path)
    pred_dict, dist_dict = eval_dataset(network, train_loader, test_loader, train_colors, test_colors)
    accuracy, n_correct, n_states = calc_accuracy(states, generators, network,
                                                  f'{model_name}, t.s. {test_size:.1f}, r.s. {random_seed}',
                                                  metric, dataset_name, pred_dict, dist_dict, eval_single_func)
    return accuracy, n_correct, n_states


def get_train_test_set(dataframe, test_size, random_seed, split_type):
    if split_type == 'adversarial':
        test_size_adv = int(round(test_size * len(dataframe.index) / 48))
        train, test = train_test_split(
            dataframe, stratify=dataframe['class_id'], test_size=test_size_adv, random_state=random_seed
        )
    elif split_type == 'ratio':
        train, test = train_test_split(
            dataframe, stratify=dataframe['distance'], test_size=test_size, random_state=random_seed
        )
    else:
        raise Exception
    return train, test


def create_loader(dataframe, solved_state_colors, test_size, random_seed, dataset_func, dataloader_cls, split_type):
    train, test = get_train_test_set(dataframe, test_size, random_seed, split_type)

    test_inputs = test['colors'].tolist()
    test_targets = test['distance'].tolist()
    test_dataset = dataset_func(test_inputs, test_targets)
    test_dataloader = dataloader_cls(test_dataset, batch_size=1, shuffle=False)

    train_inputs = train['colors'].tolist() + [solved_state_colors]
    train_targets = train['distance'].tolist() + [0]
    train_dataset = dataset_func(train_inputs, train_targets)
    train_dataloader = dataloader_cls(train_dataset, batch_size=1, shuffle=False)
    return train_dataloader, test_dataloader, train, test
