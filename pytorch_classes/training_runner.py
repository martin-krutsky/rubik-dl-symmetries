from utils.random_seed import seed_worker, seed_all, init_weights
from pytorch_classes.config import CONFIGS

from abc import ABC, abstractmethod
import ast
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)


class TrainingRunner(ABC):
    def __init__(self, model_name, model_class, model_hyperparams, lr, nr_of_epochs, config_nr,
                 loader_params, dataset_name, dataset_file, max_distance_from_goal):
        self.model = None
        self.ModelClass = model_class
        self.model_name = model_name
        self.model_hyperparams = model_hyperparams
        self.lr = lr
        self.nr_of_epochs = nr_of_epochs
        self.config = CONFIGS[config_nr]
        self.random_seed = self.config[0]
        self.test_size = self.config[1]
        self.loader_params = loader_params
        self.loader_params['worker_init_fn'] = seed_worker
        self.dataset_name = dataset_name
        self.dataset_file = dataset_file
        self.max_distance_from_goal = max_distance_from_goal

    def load_model(self, path):
        self.model = self.ModelClass(**self.model_hyperparams)
        self.model.load_state_dict(torch.load(path))
        return self.model

    def load_data(self):
        data_file_type = self.dataset_file.split('.')[-1].strip()
        if data_file_type == 'pkl':
            df = pd.read_pickle(self.dataset_file)
        elif data_file_type == 'csv':
            df = pd.read_csv(
                self.dataset_file, index_col=0, converters={'colors': ast.literal_eval, 'distance': ast.literal_eval}
            )
        else:
            raise Exception
        return df

    @abstractmethod
    def create_data_container(self, data, labels):
        pass

    @abstractmethod
    def create_data_loader(self, training_set, test_set):
        pass

    def prepare_data(self):
        df = self.load_data()

        if self.max_distance_from_goal is not None:
            df = df[df['distance'] <= self.max_distance_from_goal]

        dataset = df['colors'].tolist()
        targets = df['distance'].tolist()

        if self.test_size == 0:
            x_train, x_test, y_train, y_test = dataset, [], targets, []
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                dataset, targets, stratify=targets, test_size=self.test_size, random_state=self.random_seed
            )
        print(f'Size of trainset: {len(y_train)}, Size of testset: {len(y_test)}')

        training_set = self.create_data_container(x_train, y_train)
        test_set = self.create_data_container(x_test, y_test)

        trainloader, testloader = self.create_data_loader(training_set, test_set)
        return trainloader, testloader

    def evaluate(self, data_loader, criterion):
        self.model.eval()
        if data_loader is None:
            return None, None
        test_losses = []
        for data in tqdm(data_loader):
            with torch.no_grad():
                inputs, labels = data
                outputs = self.model(inputs)
                losses = criterion(torch.squeeze(outputs), labels.float())
                test_losses += losses.tolist()
        mean_test_loss = np.mean(test_losses)
        return test_losses, mean_test_loss

    def train(self, trainloader, testloader, verbose, print_every):
        seed_all(self.random_seed)

        self.model = self.ModelClass(**self.model_hyperparams)
        # self.model.apply(init_weights)

        criterion = nn.MSELoss()
        criterion2 = nn.L1Loss(reduction='none')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        train_losses_ls, test_losses_ls = None, None
        mean_train_losses, mean_test_losses = (
            [self.evaluate(trainloader, criterion2)[1]],
            [self.evaluate(testloader, criterion2)[1]]
        )
        print(f'Average Training MAE before training: {mean_train_losses[0]}')
        print(f'Average Test MAE before training: {mean_test_losses[0]}')

        for epoch in range(self.nr_of_epochs):
            self.model.train()
            running_loss = 0.0
            max_loss = 0.0
            train_losses_ls = []
            for i, data in tqdm(enumerate(trainloader)):
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(torch.squeeze(outputs), labels.double())
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                with torch.no_grad():
                    losses = criterion2(torch.squeeze(outputs), labels.float())
                    train_losses_ls += losses.tolist()
                    max_loss = max(max_loss, torch.max(criterion2(torch.squeeze(outputs), labels.float())))
                if verbose and (i + 1) % print_every == 0:  # print every N mini-batches
                    print(f'Epoch {epoch + 1}, Batch {i + 1}; avg loss: {running_loss / print_every:.3f}')
                    running_loss = 0.0
                    max_loss = 0.0

            mean_train_loss = np.mean(train_losses_ls)
            mean_train_losses.append(mean_train_loss)
            test_losses_ls, mean_test_loss = self.evaluate(testloader, criterion2)
            print(f'Average Test MAE in epoch {epoch + 1}: {mean_test_loss}')
            mean_test_losses.append(mean_test_loss)
        print('Finished Training')
        return train_losses_ls, mean_train_losses, test_losses_ls, mean_test_losses

    def predict(self, data, labels):
        predictions = []
        self.model.eval()
        data_container = self.create_data_container(data, labels)
        for x, y in data_container:
            with torch.no_grad():
                outputs = self.model(x)
                predictions.append(torch.squeeze(outputs))
        return predictions

    def run_pipeline(self, verbose, print_every):
        trainloader, testloader = self.prepare_data()
        train_losses_ls, mean_train_losses, test_losses_ls, mean_test_losses = self.train(
            trainloader, testloader, verbose, print_every
        )

        print(f'Average Train MAE After Training: {np.mean(train_losses_ls)}')
        print(f'Average Test MAE After Training: {np.mean(test_losses_ls)}')

        folder = f'results/{self.dataset_name}/{self.model_name}'
        os.makedirs(folder, exist_ok=True)
        rnd_test_size_suffix = f'rs{self.random_seed}_ts{self.test_size}'
        torch.save(self.model.state_dict(), f'{folder}/model_{rnd_test_size_suffix}.pth')
        np.save(f'{folder}/last_train_losses_{rnd_test_size_suffix}.npy',
                train_losses_ls)
        np.save(f'{folder}/mean_train_losses_{rnd_test_size_suffix}.npy',
                mean_train_losses)
        np.save(f'{folder}/last_test_losses_{rnd_test_size_suffix}.npy',
                test_losses_ls)
        np.save(f'{folder}/mean_test_losses_{rnd_test_size_suffix}.npy',
                mean_test_losses)
        np.save(f'{folder}/results_{rnd_test_size_suffix}.npy',
                np.array([np.mean(train_losses_ls), np.mean(test_losses_ls)]))

        folder = f'imgs/model_performance/{self.dataset_name}/ResNet'
        os.makedirs(folder, exist_ok=True)
        sns.lineplot(data=mean_train_losses, palette='orange', label='train set')
        sns.lineplot(data=mean_test_losses, palette='orange', label='test set')
        plt.xlabel("Epoch")
        plt.ylabel("Mean absolute error")
        plt.title(f"ResNet convergence plot, random seed {self.random_seed}, test size {self.test_size}")
        plt.legend()
        plt.savefig(f'{folder}/convergence_{rnd_test_size_suffix}.png')
        plt.clf()

        sns.histplot(data=train_losses_ls, bins=30)
        plt.savefig(f'{folder}/train_errors_{rnd_test_size_suffix}.png')
        plt.clf()

        sns.histplot(data=test_losses_ls, bins=30)
        plt.savefig(f'{folder}/test_errors_{rnd_test_size_suffix}.png')
        plt.clf()
