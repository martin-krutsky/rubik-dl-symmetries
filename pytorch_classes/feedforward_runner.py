from pytorch_classes.training_runner import TrainingRunner
from pytorch_classes.feedforward_model import FeedForward
from pytorch_classes.color_dataset import ColorDataset
from pytorch_classes.config import *

import sys

import torch
torch.set_default_dtype(torch.float64)


class FeedForwardTrainingRunner(TrainingRunner):
    def __init__(self, model_hyperparams, lr, nr_of_epochs, config_nr,
                 loader_params, dataset_name, dataset_file, max_distance_from_goal):
        super(FeedForwardTrainingRunner, self).__init__(
            'FeedForward', FeedForward, model_hyperparams, lr, nr_of_epochs, config_nr,
            loader_params, dataset_name, dataset_file, max_distance_from_goal
        )

    def create_data_container(self, data, labels):
        return ColorDataset(data, labels)

    def create_data_loader(self, training_set, test_set):
        trainloader = torch.utils.data.DataLoader(training_set, **self.loader_params)
        if self.test_size == 0:
            testloader = None
        else:
            testloader = torch.utils.data.DataLoader(test_set, **self.loader_params)
        return trainloader, testloader


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1].isdigit():
        config_number = int(sys.argv[1])
    else:
        config_number = 0

    resnet_runner = FeedForwardTrainingRunner(
        FEEDFORWARD_HYPERPARAMS, LEARNING_RATE, NR_OF_EPOCHS, config_number,
        RESNET_LOADER_PARAMS, DATASET_NAME, DATASET_FILE, MAX_DISTANCE
    )
    resnet_runner.run_pipeline(VERBOSE, PRINT_EVERY)
