from pytorch_classes.training_runner import TrainingRunner
from pytorch_classes.symeqnet_model import SymEqNet
from pytorch_classes.graph_dataset import create_data_list
from pytorch_classes.config import *

import sys

import torch
from torch_geometric.loader import DataLoader
torch.set_default_dtype(torch.float64)


class SymEqNetTrainingRunner(TrainingRunner):
    def __init__(self, model_name, model_hyperparams, lr, nr_of_epochs, config_nr,
                 loader_params, dataset_name, dataset_file, max_distance_from_goal):
        super(SymEqNetTrainingRunner, self).__init__(
            model_name, SymEqNet, model_hyperparams, lr, nr_of_epochs, config_nr,
            loader_params, dataset_name, dataset_file, max_distance_from_goal
        )

    def create_data_container(self, data, labels):
        return create_data_list(data, labels)

    def create_data_loader(self, training_set, test_set):
        trainloader = DataLoader(training_set, **self.loader_params)
        if self.test_size == 0:
            testloader = None
        else:
            testloader = DataLoader(test_set, **self.loader_params)
        return trainloader, testloader

    @staticmethod
    def split_input_labels(data):
        return data, data.y.squeeze()


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1].isdigit():
        config_number = int(sys.argv[1])
    else:
        config_number = 0

    resnet_runner = SymEqNetTrainingRunner(
        SYMEQNET_MODEL_NAME, SYMEQNET_HYPERPARAMS, LEARNING_RATE, NR_OF_EPOCHS,
        config_number, SYMEQNET_LOADER_PARAMS, DATASET_NAME, DATASET_FILE, MAX_DISTANCE
    )
    resnet_runner.run_pipeline(VERBOSE, PRINT_EVERY)
