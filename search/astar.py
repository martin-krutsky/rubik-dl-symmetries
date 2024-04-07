import ast
import functools
import os
import sys

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GraphLoader

from classes.cube_classes import Cube3
import pytorch_classes.graph_dataset as gd
import pytorch_classes.color_dataset as cd
from search import single_solved_rate_n_moves, eval_color_single, eval_graph_single, create_loader, \
    create_cubestate_from_gen, create_cubestate_from_colors, greedy_search_bounded, astar_search_bounded

if __name__ == '__main__':
    CURRFOLDER = '..'
    TASK = '5moves'
    MODEL = 'ResNet'
    SEARCH_TYPE = 'ASTAR'  # GREEDY/ASTAR
    
    if SEARCH_TYPE == 'ASTAR':
        search_function = astar_search_bounded
        result_dir = 'astar_search'
    elif SEARCH_TYPE == 'GREEDY':
        search_function = greedy_search_bounded
        result_dir = 'greedy_search'
    else:
        raise Exception

    if TASK == '5moves':
        FILE = '5_moves_dataset_single.pkl'
        METRIC = 'qt'
        split_type = 'ratio'
        pandas_reader = pd.read_pickle
    elif TASK in ['kociemba10', 'kociemba100', 'kociemba1000']:
        FILE = f'{TASK}_dataset.pkl'
        METRIC = 'ft'
        split_type = 'adversarial'
        pandas_reader = pd.read_pickle
    elif TASK in ['kociemba10000', 'kociemba100000']:
        FILE = f'{TASK}_dataset.csv'
        METRIC = 'ft'
        split_type = 'adversarial'
        pandas_reader = functools.partial(pd.read_csv, index_col=0,
                                          converters={'colors': ast.literal_eval,
                                                      'distance': ast.literal_eval,
                                                      'generator': ast.literal_eval})
    else:
        raise Exception

    if MODEL == 'ResNet':
        eval_func = eval_color_single
        create_loader = functools.partial(create_loader, dataset_func=cd.ColorDataset,
                                          dataloader_cls=DataLoader, split_type=split_type)
    else:
        eval_func = eval_graph_single
        create_loader = functools.partial(create_loader, dataset_func=gd.create_data_list,
                                          dataloader_cls=GraphLoader, split_type=split_type)

    CONFIGS = [(rnd_seed, test_size) for test_size in np.arange(0.1, 1.0, 0.1) for rnd_seed in range(0, 10)]
    config_nr = 89  # int(sys.argv[1])
    rnd_seed, tst_size = CONFIGS[config_nr]

    df = pandas_reader(f'{CURRFOLDER}/data/processed/{FILE}')
    df['colors'] = df['colors'].map(np.array)
    if TASK in ['kociemba10000', 'kociemba100000']:
        df['state'] = df['generator'].map(create_cubestate_from_gen)  # df['colors'].map(create_cubestate_from_colors)
    df = df[df['distance'] <= 3]
    solved_colors = (Cube3().generate_goal_states(1)[0].colors // 9).tolist()
    tr_loader, ts_loader, train_df, test_df = create_loader(df, solved_colors, tst_size, rnd_seed)
    solved_rt, perf_solved_rt, solved_nr, perfectly_solved_nr, paths_or_expansions, nr_of_states = single_solved_rate_n_moves(
        df.state, df.generator, train_df.colors, test_df.colors.tolist() + [solved_colors],
        tr_loader, ts_loader, METRIC, TASK, MODEL, tst_size, rnd_seed, eval_func, CURRFOLDER,
        search_function
    )
    print(f'Solved rate: {solved_rt}, Perf. solved rate: {perf_solved_rt}')
    out_folder = f'{CURRFOLDER}/data/{result_dir}/{TASK}/{MODEL}'
    os.makedirs(out_folder, exist_ok=True)
    with open(f'{out_folder}/ts{tst_size:.1f}_rs{rnd_seed}.txt', 'w') as f:
        if SEARCH_TYPE == 'ASTAR':
            mean_expansion_rate = np.mean(paths_or_expansions)
            median_expansion_rate = np.median(paths_or_expansions)
            f.write('\n'.join([
                str(solved_rt), str(perf_solved_rt), str(solved_nr), str(perfectly_solved_nr),
                str(mean_expansion_rate), str(median_expansion_rate), str(nr_of_states)
            ]))
        else:
            f.write('\n'.join([
                str(solved_rt), str(perf_solved_rt), str(solved_nr), str(perfectly_solved_nr), str(nr_of_states)
            ]))
