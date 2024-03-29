import numpy as np


NR_OF_RND_SEEDS = 10
TRAIN_TEST_SPLIT_TYPE = 'ratio'  # ratio / adversarial
if TRAIN_TEST_SPLIT_TYPE == 'ratio':
    TEST_SIZES = np.arange(0.1, 1.0, 0.1)
elif TRAIN_TEST_SPLIT_TYPE == 'adversarial':
    TEST_SIZES = [48 - i for i in range(1, 11)]
CONFIGS = [(rnd_seed, test_size) for test_size in TEST_SIZES for rnd_seed in range(0, NR_OF_RND_SEEDS)]

NR_OF_EPOCHS = 100
VERBOSE = True
CUBE_SIDE_SIZE = 3

GRAPH_LOADER_PARAMS = {
    'batch_size': 1024,
    'shuffle': True,
    'num_workers': 0,
}
COLOR_LOADER_PARAMS = {
    'batch_size': 1024,
    'shuffle': True,
    'num_workers': 0,
}
STATE_DIM = (CUBE_SIDE_SIZE ** 2) * 6


SYMEQNET_HYPERPARAMS = {
    'hidden_graph_channels': 9,
    'h1_dim': 500,
    'resnet_dim': 100,
    'num_resnet_blocks': 2,
    'batch_norm': True,
    'gnn_layer_class': 'GENConv'
}

RESNET_HYPERPARAMS = {
    'state_dim': STATE_DIM,
    'one_hot_depth': 6,
    'h1_dim': 500,
    'resnet_dim': 100,
    'num_resnet_blocks': 2,
    'out_dim': 1, 
    'batch_norm': True
}
FEEDFORWARD_HYPERPARAMS = {
    'state_dim': STATE_DIM,
    'one_hot_depth': 6,
    'h_dim': 100,
    'out_dim': 1, 
}

KOCIEMBAS_SIZE = 100000
RESNET_MODEL_NAME = f'ResNet'
SYMEQNET_MODEL_NAME = f'GENNet'
FEEDFORWARD_MODEL_NAME = 'FeedForward'

LEARNING_RATE = 0.001

MAX_DISTANCE = 5
DATASET_NAME = f'5moves'  # 5moves/6moves/kociemba/kociemba{KOCIEMBAS_SIZE}
# 5_moves_dataset_single.pkl/6_moves_dataset_single.csv/kociemba_dataset.csv/kociemba{KOCIEMBAS_SIZE}_dataset.pkl
DATASET_FILE = f'data/processed/5_moves_dataset_single.pkl'

PRINT_EVERY = 100

POS_ARRAY = np.array([
    # 1
    [0.5, 0.5, 0], [1.5, 0.5, 0], [2.5, 0.5, 0],
    [0.5, 1.5, 0], [1.5, 1.5, 0], [2.5, 1.5, 0],
    [0.5, 2.5, 0], [1.5, 2.5, 0], [2.5, 2.5, 0],

    # 2
    [2.5, 0.5, 3], [1.5, 0.5, 3], [0.5, 0.5, 3],
    [2.5, 1.5, 3], [1.5, 1.5, 3], [0.5, 1.5, 3],
    [2.5, 2.5, 3], [1.5, 2.5, 3], [0.5, 2.5, 3],

    # 3
    [2.5, 0, 2.5], [2.5, 0, 1.5], [2.5, 0, 0.5],
    [1.5, 0, 2.5], [1.5, 0, 1.5], [1.5, 0, 0.5],
    [0.5, 0, 2.5], [0.5, 0, 1.5], [0.5, 0, 0.5],

    # 4
    [0.5, 3, 2.5], [0.5, 3, 1.5], [0.5, 3, 0.5],
    [1.5, 3, 2.5], [1.5, 3, 1.5], [1.5, 3, 0.5],
    [2.5, 3, 2.5], [2.5, 3, 1.5], [2.5, 3, 0.5],

    # 5
    [3, 2.5, 2.5], [3, 2.5, 1.5], [3, 2.5, 0.5],
    [3, 1.5, 2.5], [3, 1.5, 1.5], [3, 1.5, 0.5],
    [3, 0.5, 2.5], [3, 0.5, 1.5], [3, 0.5, 0.5],

    # 6
    [0, 0.5, 2.5], [0, 0.5, 1.5], [0, 0.5, 0.5],
    [0, 1.5, 2.5], [0, 1.5, 1.5], [0, 1.5, 0.5],
    [0, 2.5, 2.5], [0, 2.5, 1.5], [0, 2.5, 0.5],
])
