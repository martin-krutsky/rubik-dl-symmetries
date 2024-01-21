import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn.pool import global_mean_pool


class GCNNet(torch.nn.Module):
    def __init__(self, hidden_graph_channels, hidden_lin_channels=10):
        super(GCNNet, self).__init__()
        self.conv1 = gnn.SimpleConv()
        self.lin1 = nn.Linear(hidden_graph_channels, hidden_lin_channels)
        self.out = nn.Linear(hidden_lin_channels, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = global_mean_pool(x, data.batch if data.batch is not None else torch.zeros((data.x.size(0)), dtype=torch.int64))
        x = F.relu(x)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.out(x)
        return x