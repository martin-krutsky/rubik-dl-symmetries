from pytorch_classes.training_runner import TrainingRunner
from pytorch_classes.resnet_model import ResNet
from pytorch_classes.color_dataset import ColorDataset
from pytorch_classes.config import *

import sys

import torch
torch.set_default_dtype(torch.float64)


class ResNetTrainingRunner(TrainingRunner):
    def __init__(self, model_name, model_hyperparams, lr, nr_of_epochs, config_nr,
                 loader_params, dataset_name, dataset_file, max_distance_from_goal):
        super(ResNetTrainingRunner, self).__init__(
            model_name, ResNet, model_hyperparams, lr, nr_of_epochs, config_nr,
            loader_params, dataset_name, dataset_file, max_distance_from_goal
        )

    def create_data_container(self, data, labels):
        return ColorDataset(data, labels)


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1].isdigit():
        config_number = int(sys.argv[1])
    else:
        config_number = 0

    resnet_runner = ResNetTrainingRunner(
        RESNET_MODEL_NAME, RESNET_HYPERPARAMS, LEARNING_RATE, NR_OF_EPOCHS,
        config_number, RESNET_LOADER_PARAMS, DATASET_NAME, DATASET_FILE, MAX_DISTANCE
    )
    resnet_runner.run_pipeline(VERBOSE, PRINT_EVERY)
