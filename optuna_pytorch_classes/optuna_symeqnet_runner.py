from utils.random_seed import seed_all
from pytorch_classes.symeqnet_model import SymEqNet
from pytorch_classes.symeqnet_runner import SymEqNetTrainingRunner
from pytorch_classes.config import *

import sys
from tqdm import tqdm

import mlflow
import mlflow.pytorch
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

config_number = 10


def define_model(trial: optuna.Trial):
    num_resnet_blocks = trial.suggest_int('n_resnet_blocks', 0, 4)
    # print(f'Nr of resnet blocks: {num_resnet_blocks}')
    batch_norm = trial.suggest_categorical('batch_normalization', [True, False])
    # print(f'batch normalization: {batch_norm}')
    gnn_layer_class = trial.suggest_categorical(
        'gnn_layer_class', ['PNAConv', 'GeneralConv', 'GATv2Conv']  # TransformerConv
    )
    # print(f'GNN layer class: {gnn_layer_class}')
    graph_channels = trial.suggest_int('n_graph_channels', 50, 500, 50)
    # print(f'Nr of graph output channels: {gnn_layer_class}')
    hidden_layer_neurons = trial.suggest_int('n_hidden_neurons', 50, 500, 50)
    # print(f'Nr of hidden layer neurons: {hidden_layer_neurons}')

    other_kwds = {}
    if gnn_layer_class == 'PNAConv':
        other_kwds['aggregators'] = ['mean', 'min', 'max', 'std']
        other_kwds['scalers'] = ['identity', 'amplification', 'attenuation']
        other_kwds['deg'] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1])
        other_kwds['towers'] = trial.suggest_categorical('towers', [1, 2, 5])
        other_kwds['pre_layers'] = trial.suggest_int('pre_layers', 1, 4)
        other_kwds['post_layers'] = trial.suggest_int('post_layers', 1, 4)
    elif gnn_layer_class == 'GeneralConv':
        other_kwds['aggr'] = trial.suggest_categorical('aggr', ["add", "mean", "max"])
        other_kwds['skip_linear'] = trial.suggest_categorical('skip_linear', [True, False])
        other_kwds['heads'] = trial.suggest_int('heads', 1, 5)
        other_kwds['attention'] = trial.suggest_categorical('attention', [True, False])
        if other_kwds['attention']:
            other_kwds['attention_type'] = trial.suggest_categorical('attention_type', ["additive", "dot_product"])
        other_kwds['l2_normalize'] = trial.suggest_categorical('l2_normalize', [True, False])
    elif gnn_layer_class == 'GATv2Conv':
        other_kwds['heads'] = trial.suggest_int('heads', 1, 5)
        other_kwds['concat'] = trial.suggest_categorical('concat', [True, False])
        other_kwds['negative_slope '] = trial.suggest_float('negative_slope', 0, 0.3)
        other_kwds['dropout'] = trial.suggest_float('dropout', 0, 0.5)
        other_kwds['share_weights'] = trial.suggest_categorical('share_weights', [True, False])
    else:
        raise Exception(f'Unsupported GNN layer ({gnn_layer_class}) for hyperparameter choice!')

    return SymEqNet(gnn_layer_class=gnn_layer_class, hidden_graph_channels=graph_channels,
                    hidden_lin_channels=hidden_layer_neurons, num_resnet_blocks=num_resnet_blocks,
                    batch_norm=batch_norm, other_kwds=other_kwds)


def objective(trial: optuna.Trial, runner, trainloader, testloader):
    with mlflow.start_run():
        seed_all(runner.random_seed)
        model = define_model(trial)
        runner.model = model

        criterion = nn.MSELoss()
        criterion2 = nn.L1Loss(reduction='none')

        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
        mlflow.log_params(trial.params)

        train_losses_ls, test_losses_ls = None, None
        mean_train_losses, mean_test_losses = (
            [runner.evaluate(trainloader, criterion2)[1]],
            [runner.evaluate(testloader, criterion2)[1]]
        )
        print(f'Average Training MAE before training: {mean_train_losses[0]}')
        print(f'Average Test MAE before training: {mean_test_losses[0]}')

        for epoch in range(runner.nr_of_epochs):
            model.train()
            running_loss = 0.0
            max_loss = 0.0
            train_losses_ls = []
            for i, data in enumerate(trainloader):
                inputs, labels = runner.split_input_labels(data)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(torch.squeeze(outputs), labels.double())
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                with torch.no_grad():
                    losses = criterion2(torch.squeeze(outputs), labels.float())
                    train_losses_ls += losses.tolist()
                    max_loss = max(max_loss, torch.max(criterion2(torch.squeeze(outputs), labels.float())))
                if VERBOSE and (i + 1) % PRINT_EVERY == 0:  # print every N mini-batches
                    print(f'Epoch {epoch + 1}, Batch {i + 1}; avg loss: {running_loss / PRINT_EVERY:.3f}')
                    running_loss = 0.0
                    max_loss = 0.0

            mean_train_loss = np.mean(train_losses_ls)
            mean_train_losses.append(mean_train_loss)

            test_losses_ls, mean_test_loss = runner.evaluate(testloader, criterion2)
            print(f'Average Test MAE in epoch {epoch + 1}: {mean_test_loss}')
            mean_test_losses.append(mean_test_loss)
            trial.report(float(mean_test_loss), epoch)

            mlflow.log_metric("mean_train_loss", float(mean_train_loss), step=epoch)
            mlflow.log_metric("mean_test_loss", float(mean_test_loss), step=epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    mlflow.end_run()

    print(f'Average Train MAE After Training: {np.mean(train_losses_ls)}')
    print(f'Average Test MAE After Training: {np.mean(test_losses_ls)}')

    return np.mean(test_losses_ls)


def run():
    runner = SymEqNetTrainingRunner(
        SYMEQNET_MODEL_NAME, SYMEQNET_HYPERPARAMS, LEARNING_RATE, NR_OF_EPOCHS,
        config_number, GRAPH_LOADER_PARAMS, DATASET_NAME, DATASET_FILE, MAX_DISTANCE
    )
    trainloader, testloader = runner.prepare_data()

    objective_partial = lambda trial: objective(trial, runner, trainloader, testloader)
    study = optuna.create_study(direction='minimize')

    study.optimize(objective_partial, n_jobs=8, n_trials=1000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: ", best_trial.value)

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == '__main__':
    run()
