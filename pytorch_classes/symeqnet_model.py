import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn.pool import global_mean_pool


class SymEqNet(torch.nn.Module):
    def __init__(self, gnn_layer_class='SimpleConv', node_features_size=9, hidden_graph_channels=10,
                 h1_dim=10, resnet_dim=10, num_resnet_blocks=1, batch_norm=True, other_kwds=None):
        super(SymEqNet, self).__init__()
        if other_kwds is None:
            other_kwds = {}
        self.num_resnet_blocks = num_resnet_blocks
        self.batch_norm = batch_norm

        # GATConv, PNAConv, GeneralConv, TransformerConv, GATv2Conv
        if gnn_layer_class == 'SimpleConv':
            self.conv1 = gnn.SimpleConv()
        elif gnn_layer_class == 'GATConv':
            self.conv1 = gnn.GATConv(in_channels=node_features_size, out_channels=hidden_graph_channels,
                                     edge_dim=1, **other_kwds)
        elif gnn_layer_class == 'PNAConv':
            self.conv1 = gnn.PNAConv(in_channels=node_features_size, out_channels=hidden_graph_channels,
                                     edge_dim=1, **other_kwds)
        elif gnn_layer_class == 'GeneralConv':
            self.conv1 = gnn.GeneralConv(in_channels=node_features_size, out_channels=hidden_graph_channels,
                                         in_edge_channels=1, **other_kwds)
        elif gnn_layer_class == 'TransformerConv':
            self.conv1 = gnn.TransformerConv(in_channels=node_features_size, out_channels=hidden_graph_channels,
                                             edge_dim=1, **other_kwds)
        elif gnn_layer_class == 'GATv2Conv':
            self.conv1 = gnn.GATv2Conv(in_channels=node_features_size, out_channels=hidden_graph_channels,
                                       edge_dim=1, **other_kwds)
        elif gnn_layer_class == 'GENConv':
            self.conv1 = gnn.GENConv(in_channels=node_features_size, out_channels=hidden_graph_channels,
                                     edge_dim=1, aggr='add', **other_kwds)
        else:
            raise Exception(f'Unsupported GNN layer ({gnn_layer_class}) name!')

        self.fc1 = nn.Linear(hidden_graph_channels, h1_dim)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(h1_dim)
        self.fc2 = nn.Linear(h1_dim, resnet_dim)
        if self.batch_norm:
            self.bn2 = nn.BatchNorm1d(resnet_dim)
        
        if self.num_resnet_blocks > 0:
            self.blocks = nn.ModuleList()
            for block_num in range(self.num_resnet_blocks):
                if self.batch_norm:
                    res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                    res_bn1 = nn.BatchNorm1d(resnet_dim)
                    res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                    res_bn2 = nn.BatchNorm1d(resnet_dim)
                    self.blocks.append(nn.ModuleList([res_fc1, res_bn1, res_fc2, res_bn2]))
                else:
                    res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                    res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                    self.blocks.append(nn.ModuleList([res_fc1, res_fc2]))
                    
        else:
            self.fc3 = nn.Linear(resnet_dim)
            if self.batch_norm:
                self.bn3 = nn.BatchNorm1d(resnet_dim)
        
        self.fc_out = nn.Linear(resnet_dim, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = global_mean_pool(x, data.batch if data.batch is not None else torch.zeros((data.x.size(0)), dtype=torch.int64))
        
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        res_input = x
        
        if self.num_resnet_blocks > 0:
            for block_num in range(self.num_resnet_blocks):
                res_input = x
                if self.batch_norm:
                    x = self.blocks[block_num][0](x)
                    x = self.blocks[block_num][1](x)
                    x = F.relu(x)
                    x = self.blocks[block_num][2](x)
                    x = self.blocks[block_num][3](x)
                    x = F.relu(x + res_input)
                else:
                    x = self.blocks[block_num][0](x)
                    x = F.relu(x)
                    x = self.blocks[block_num][1](x)
                    x = F.relu(x + res_input)
        else:
            x = self.fc3(x)
            if self.batch_norm:
                x = self.bn3(x)
            x = F.relu(x)
        
        x = self.fc_out(x)
        return x