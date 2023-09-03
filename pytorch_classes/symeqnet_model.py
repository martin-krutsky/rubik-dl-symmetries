import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn.pool import global_mean_pool


class SymEqNet(torch.nn.Module):
    def __init__(self, node_features_size=9, hidden_graph_channels=10, hidden_lin_channels=10, num_resnet_blocks=1):
        super(SymEqNet, self).__init__()
        self.num_resnet_blocks = num_resnet_blocks
        self.conv1 = gnn.GATConv(in_channels=node_features_size, out_channels=hidden_graph_channels, edge_dim=1)
        self.bn1 = nn.BatchNorm1d(hidden_graph_channels)
        self.fc2 = nn.Linear(hidden_graph_channels, hidden_lin_channels)
        self.bn2 = nn.BatchNorm1d(hidden_lin_channels)
        
        self.blocks = nn.ModuleList()
        for block_num in range(self.num_resnet_blocks):
            res_fc1 = nn.Linear(hidden_lin_channels, hidden_lin_channels)
            res_bn1 = nn.BatchNorm1d(hidden_lin_channels)
            res_fc2 = nn.Linear(hidden_lin_channels, hidden_lin_channels)
            res_bn2 = nn.BatchNorm1d(hidden_lin_channels)
            self.blocks.append(nn.ModuleList([res_fc1, res_bn1, res_fc2, res_bn2]))
        self.fc_out = nn.Linear(hidden_lin_channels, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = global_mean_pool(x, data.batch if data.batch is not None else torch.zeros((data.x.size(0)), dtype=torch.int64))
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        res_input = x
        
        for block_num in range(self.num_resnet_blocks):
            res_input = x
            x = self.blocks[block_num][0](x)
            x = self.blocks[block_num][1](x)
            x = F.relu(x)
            x = self.blocks[block_num][2](x)
            x = self.blocks[block_num][3](x)
            x = F.relu(x + res_input)
        
        x = self.fc_out(x)
        return x