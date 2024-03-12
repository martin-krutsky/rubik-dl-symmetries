import ast
import functools
import os
import sys

import pandas as pd
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GraphLoader

from classes.cube_classes import Cube3
from pytorch_classes.config import CONFIGS
import pytorch_classes.graph_dataset as gd
import pytorch_classes.color_dataset as cd
from search.search import single_accuracy_n_moves, eval_color_single, eval_graph_single, create_loader

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
                                          converters={'colors': ast.literal_eval,
                                                      'distance': ast.literal_eval})
    else:
        raise Exception

    if MODEL == 'ResNet':
        eval_func = eval_color_single
        create_loader = functools.partial(create_loader, dataset_func=cd.ColorDataset, dataloader_cls=DataLoader)
    else:
        eval_func = eval_graph_single
        create_loader = functools.partial(create_loader, dataset_func=gd.create_data_list, dataloader_cls=GraphLoader)

    CONFIG_NR = int(sys.argv[1])
    rnd_seed, tst_size = CONFIGS[CONFIG_NR]

    df = pandas_reader(f'{CURRFOLDER}/data/processed/{FILE}')
    # df = df[df['distance'] <= 1]
    solved_colors = (Cube3().generate_goal_states(1)[0].colors // 9).tolist()
    tr_loader, ts_loader, train_df, test_df = create_loader(df, solved_colors, tst_size, rnd_seed)
    acc, correct, nr_of_states = single_accuracy_n_moves(df.state, df.generator,
                                                         train_df.colors, test_df.colors.tolist() + [solved_colors],
                                                         tr_loader, ts_loader, METRIC, TASK, MODEL, tst_size, rnd_seed,
                                                         eval_func, CURRFOLDER)
    print(f'Accuracy: {acc}')
    out_folder = f'{CURRFOLDER}/data/evals/{TASK}/{MODEL}'
    os.makedirs(out_folder, exist_ok=True)
    with open(f'{out_folder}/ts{tst_size:.1f}_rs{rnd_seed}.txt', 'w') as f:
        f.write('\n'.join([str(acc), str(correct), str(nr_of_states)]))