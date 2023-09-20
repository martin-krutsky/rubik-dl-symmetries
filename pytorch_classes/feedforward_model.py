import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, state_dim: int, one_hot_depth: int, h_dim: int, out_dim: int):
        super().__init__()
        self.one_hot_depth: int = one_hot_depth
        self.state_dim: int = state_dim

        self.fc1 = nn.Linear(one_hot_depth*state_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc_out = nn.Linear(h_dim, out_dim)

    def forward(self, states_nnet):
        x = states_nnet
        x = F.one_hot(x.long(), self.one_hot_depth)
        x = x.double()
        x = x.view(-1, self.state_dim * self.one_hot_depth)

        # first two hidden layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        # output
        x = self.fc_out(x)
        return x
