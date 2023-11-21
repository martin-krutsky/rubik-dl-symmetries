import numpy as np


NR_OF_RND_SEEDS = 10
TRAIN_TEST_SPLIT_TYPE = 'ratio'  # ratio / adversarial
if TRAIN_TEST_SPLIT_TYPE == 'ratio':
    TEST_SIZES = np.arange(0.1, 1.0, 0.1)
elif TRAIN_TEST_SPLIT_TYPE == 'adversarial':
    TEST_SIZES = [48 - i for i in range(1, 11)]
CONFIGS = [(rnd_seed, test_size) for rnd_seed in range(0, NR_OF_RND_SEEDS) for test_size in TEST_SIZES]

NR_OF_EPOCHS = 10
VERBOSE = True
CUBE_SIDE_SIZE = 3

SYMEQNET_LOADER_PARAMS = {
    'batch_size': 1024,
    'shuffle': True,
    'num_workers': 0,
}
RESNET_LOADER_PARAMS = {
    'batch_size': 1024,
    'shuffle': True,
    'num_workers': 8,
}
STATE_DIM = (CUBE_SIDE_SIZE ** 2) * 6


SYMEQNET_HYPERPARAMS = {
    'hidden_graph_channels': 100,
    'hidden_lin_channels': 100,
    'num_resnet_blocks': 2,
    'batch_norm': True
}

RESNET_HYPERPARAMS = {
    'state_dim': STATE_DIM,
    'one_hot_depth': 6,
    'h1_dim': 100,
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

RESNET_MODEL_NAME = 'ResNet'
FEEDFORWARD_MODEL_NAME = 'FeedForward'
SYMEQNET_MODEL_NAME = 'SymEqNet_Transformer'

LEARNING_RATE = 0.001

MAX_DISTANCE = 3
DATASET_NAME = '5moves'  # 5moves/6moves/kociemba
# 5_moves_dataset_single.pkl/6_moves_dataset_single.csv/kociemba_dataset.csv
DATASET_FILE = 'data/processed/5_moves_dataset_single.pkl'

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
