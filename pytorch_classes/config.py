import numpy as np


NR_OF_RND_SEEDS = 10
CONFIGS = [(rnd_seed, test_size) for rnd_seed in range(0, NR_OF_RND_SEEDS) for test_size in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]]


NR_OF_EPOCHS = 10
VERBOSE = True
CUBE_SIDE_SIZE = 3

SYMEQNET_LOADER_PARAMS = {
    'batch_size': 1024,
    'shuffle': True,
    'num_workers': 0,
}
SYMEQNET_HYPERPARAMS = {
    'hidden_graph_channels': 50,
    'hidden_lin_channels': 100,
    'num_resnet_blocks': 4
}

RESNET_LOADER_PARAMS = {
    'batch_size': 1024,
    'shuffle': True,
    'num_workers': 8,
}
STATE_DIM = (CUBE_SIDE_SIZE ** 2) * 6
RESNET_HYPERPARAMS = {
    'state_dim': STATE_DIM,
    'one_hot_depth': 6,
    'h1_dim': 5000,
    'resnet_dim': 1000,
    'num_resnet_blocks': 4,
    'out_dim': 1, 
    'batch_norm': True
}

LEARNING_RATE = 0.001


DATASET_NAME = '4moves'  # 5moves/6moves/kociemba
# 5_moves_dataset_single.pkl/6_moves_dataset_single.csv/kociemba_dataset.csv
DATASET_FILE = 'data/processed/5_moves_dataset_single.pkl'
DATASET_TYPE = 'pkl'  # pkl/csv

PRINT_EVERY = 100

POS_ARRAY = np.array([
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