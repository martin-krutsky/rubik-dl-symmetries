import ast
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
torch.set_default_dtype(torch.float64)

from classes.cube_classes import Cube3State, Cube3
from utils.random_seed import seed_worker, seed_all, init_weights
from pytorch_classes.symeqnet_model import SymEqNet
from pytorch_classes.symeqnet_dataset import create_data_list
from pytorch_classes.config import *


if len(sys.argv) >= 2 and sys.argv[1].isdigit():
    config_nr = int(sys.argv[1])
else:
    config_nr = 0
CURR_CONFIG = CONFIGS[config_nr]

RANDOM_SEED = CURR_CONFIG[0]
TEST_SIZE = CURR_CONFIG[1]
SYMEQNET_LOADER_PARAMS['worker_init_fn'] = seed_worker


def load_model(PATH):
    model = SymEqNet(**SYMEQNET_HYPERPARAMS)
    model.load_state_dict(torch.load(PATH))
    return model


def predict(model, X, labels):
    predictions = []
    model.eval()
    data_list = create_data_list(X, labels)
    for x, y in data_list:
        with torch.no_grad():
            outputs = network(x)
            predictions.append(torch.squeeze(outputs))
    return predictions


def eval_test(network, epoch, test_loader, test_criterion):
    print('Evaluating on test dataset')
    network.eval()
    if test_loader is None:
        return None, None
    test_losses = []
    for data in tqdm(test_loader):
        with torch.no_grad():
            labels = data.y.squeeze()
            outputs = network(data)
            losses = test_criterion(torch.squeeze(outputs), labels.float())
            test_losses += losses.tolist()
    mean_test_loss = np.mean(test_losses)
    print(f'Average Test MAE in epoch {epoch+1}: {mean_test_loss}')
    return test_losses, mean_test_loss


def prepare_data(rnd_seed, test_size, max_distance=None):
    if DATASET_TYPE == 'pkl':
        df = pd.read_pickle(DATASET_FILE)
    elif DATASET_TYPE == 'csv':
        df = pd.read_csv(DATASET_FILE, index_col=0,
                         converters={'colors': ast.literal_eval, 'distance': ast.literal_eval})
    else:
        raise Exception
    
    if max_distance is not None:
        df = df[df['distance'] <= max_distance]
    dataset = df['colors'].tolist()
    class_ids = df['class_id'].tolist()
    targets = df['distance'].tolist()

    if test_size == 0:
        X_train, X_test, y_train, y_test = dataset, [], targets, []
    else:
        X_train, X_test, y_train, y_test = train_test_split(dataset, targets, stratify=targets, test_size=test_size, random_state=rnd_seed)
    print(f'Size of trainset: {len(y_train)}, Size of testset: {len(y_test)}')

    training_set = create_data_list(X_train, y_train)
    test_set = create_data_list(X_test, y_test)

    trainloader = DataLoader(training_set, **SYMEQNET_LOADER_PARAMS)
    if test_size == 0:
        testloader = None
    else:
        testloader = DataLoader(test_set, **SYMEQNET_LOADER_PARAMS)
    return trainloader, testloader


def train(trainloader, testloader, rnd_seed):
    seed_all(rnd_seed)
    
    net = SymEqNet(**SYMEQNET_HYPERPARAMS)
    # net.apply(init_weights)

    criterion = nn.MSELoss()
    criterion2 = nn.L1Loss(reduction='none')
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    train_losses_ls = None
    mean_train_losses, mean_test_losses = (
        [eval_test(net, -1, trainloader, criterion2)[1]],
        [eval_test(net, -1, testloader, criterion2)[1]]
    )
    for epoch in range(NR_OF_EPOCHS):
        print(f'Training epoch {epoch+1}')
        net.train()
        running_loss = 0.0
        max_loss = 0.0
        train_losses_ls = []
        for i, data in tqdm(enumerate(trainloader)):
            labels = data.y.squeeze()
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(data)
            loss = criterion(torch.squeeze(outputs), labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            with torch.no_grad():
                losses = criterion2(torch.squeeze(outputs), labels.float())
                train_losses_ls += losses.tolist()
                max_loss = max(
                    max_loss, torch.max(criterion2(torch.squeeze(outputs), labels.float()))
                )
            if VERBOSE and (i+1) % PRINT_EVERY == 0:    # print every N mini-batches
                print(f'Epoch {epoch + 1}, Batch {i + 1}; avg loss: {running_loss/PRINT_EVERY:.3f}')
                running_loss = 0.0
                max_loss = 0.0
        
        mean_train_loss = np.mean(train_losses_ls)
        mean_train_losses.append(mean_train_loss)
        test_losses_ls, mean_test_loss = eval_test(net, epoch, testloader, criterion2)
        mean_test_losses.append(mean_test_loss)
    print('Finished Training')
    return net, train_losses_ls, mean_train_losses, test_losses_ls, mean_test_losses

    
def run_pipeline(rnd_seed=RANDOM_SEED, test_size=TEST_SIZE, max_distance=None):
    trainloader, testloader = prepare_data(rnd_seed, test_size, max_distance=max_distance)
    trained_net, train_losses_ls, mean_train_losses, test_losses_ls, mean_test_losses = train(
        trainloader, testloader, rnd_seed
    )
    
    print(f'Average Train MAE After Training: {np.mean(train_losses_ls)}')
    print(f'Average Test MAE After Training: {np.mean(test_losses_ls)}')
    
    folder = f'results/{DATASET_NAME}/SymEqNet'
    os.makedirs(folder, exist_ok=True)
    torch.save(trained_net.state_dict(), f'{folder}/model_rs{RANDOM_SEED}_ts{test_size}.pth')
    np.save(f'{folder}/last_train_losses_rs{rnd_seed}_ts{test_size}.npy',
            train_losses_ls)
    np.save(f'{folder}/mean_train_losses_rs{rnd_seed}_ts{test_size}.npy',
            mean_train_losses)
    np.save(f'{folder}/last_test_losses_rs{rnd_seed}_ts{test_size}.npy',
            test_losses_ls)
    np.save(f'{folder}/mean_test_losses_rs{rnd_seed}_ts{test_size}.npy',
            mean_test_losses)
    np.save(f'{folder}/results_rs{rnd_seed}_ts{test_size}.npy',
            np.array([np.mean(train_losses_ls), np.mean(test_losses_ls)]))
    
    folder = f'imgs/model_performance/{DATASET_NAME}/SymEqNet'
    os.makedirs(folder, exist_ok=True)
    sns.lineplot(data=mean_train_losses, palette='orange', label='train set')
    sns.lineplot(data=mean_test_losses, palette='blue', label='test set')
    plt.xlabel("Epoch")
    plt.ylabel("Mean absolute error")
    plt.title(f"SymEqNet convergence plot, random seed {rnd_seed}, test size {test_size}")
    plt.legend()
    plt.savefig(f'{folder}/convergence_rs{rnd_seed}_ts{test_size}.png')
    plt.clf()

    sns.histplot(data=train_losses_ls, bins=30)
    plt.savefig(f'{folder}/train_errors_rs{rnd_seed}_ts{test_size}.png')
    plt.clf()

    sns.histplot(data=test_losses_ls, bins=30)
    plt.savefig(f'{folder}/test_errors_rs{rnd_seed}_ts{test_size}.png')
    plt.clf()

    
if __name__ == '__main__':
    run_pipeline(max_distance=MAX_DISTANCE)
