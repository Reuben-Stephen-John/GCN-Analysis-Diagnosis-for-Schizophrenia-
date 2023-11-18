import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GraphConv, GatedGraphConv, GravNetConv, GATConv, GATv2Conv, SuperGATConv, BatchNorm
from torch_geometric.nn import SAGEConv


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, number_of_features, number_of_classes):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(number_of_features, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        self.lin = Linear(hidden_channels, number_of_classes)

    def forward(self, x, edge_index, batch):
        ds = 0.65
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.bn1(x)
        x = F.dropout(x, p=ds, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.bn2(x)
        x = F.dropout(x, p=ds, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.dropout(x, p=ds, training=self.training)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=ds, training=self.training)
        x = self.lin(x)

        return x

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, number_of_features, number_of_classes):
        super(GAT, self).__init__()
        out_dimension = hidden_channels
        self.ds = 0.2
        heads1 = 56  # 24 worked nice
        heads2 = 56
        in_dimension = out_dimension * heads1
        lin_dimension = hidden_channels
        self.conv1 = GATConv(in_channels=number_of_features, out_channels=out_dimension, heads=heads1, dropout=self.ds)
        self.bn1 = BatchNorm(hidden_channels*heads1)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(in_channels=in_dimension, out_channels=out_dimension, heads=heads2, concat=False,
                             dropout=self.ds)
        self.bn2 = BatchNorm(out_dimension)
        self.lin = Linear(lin_dimension, number_of_classes)

    def forward(self, x, edge_index, batch):
        ds = 0.5
        x = F.elu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=ds, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.dropout(x, p=ds, training=self.training)

        x = global_mean_pool(x, batch)

        # 3. Apply a final classifier
        x = F.dropout(x, p=ds, training=self.training)
        x = self.lin(x)

        return x


class SAGENET(torch.nn.Module):
    def __init__(self, hidden_channels, number_of_features, number_of_classes):
        super(SAGENET, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = SAGEConv(number_of_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)

        self.lin = Linear(hidden_channels, number_of_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x=x, edge_index=edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
    
class LGM(nn.Module):
    def __init__(self, number_of_features, number_of_classes, hidden_channels):
        super().__init__()
        self.lin1 = nn.Linear(number_of_features, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, number_of_classes)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.bn1(x)
        x = F.relu(self.lin2(x))
        return x
    
# https://github.com/rusty1s/pytorch_geometric/blob/master/examples