import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch_geometric.nn import GCNConv, GATConv, APPNP, GINConv, global_mean_pool, JumpingKnowledge
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_remaining_self_loops
from torch_sparse import spmm
from sparse_smoothing.utils import to_undirected


class SparseGCNConv(GCNConv):
    def __init__(self, in_channel, out_channel, **kwargs):
        super().__init__(in_channels=in_channel, out_channels=out_channel, **kwargs)

    def forward(self, x, edge_idx, n, d):
        x = spmm(x, torch.ones_like(x[0]), n, d, self.weight)
        edge_idx, norm = gcn_norm(edge_idx, None, x.size(0), self.improved, self.add_self_loops, x.dtype)
        return self.propagate(edge_idx, x=x, edge_weight=norm, size=None)



class SparseGATConv(GATConv):
    def __init__(self, in_channel, out_channel, **kwargs):
        super().__init__(in_channels=in_channel, out_channels=out_channel, **kwargs)

    def forward(self, x, edge_idx, n, d):
        edge_idx, _ = add_remaining_self_loops(edge_idx)
        x = spmm(x, torch.ones_like(x[0]), n, d, self.weight)
        return self.propagate(edge_idx, x=x)


class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Our fan_in is interpreted by PyTorch as fan_out (swapped dimensions)
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, index, value, n):
        res = spmm(index, value, n, self.in_features, self.weight)
        if self.bias is not None:
            res += self.bias[None, :]
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None)


class GCN(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden, p_dropout=0.5):
        super().__init__()
        self.conv1 = SparseGCNConv(n_features, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_classes)
        self.p_dropout = p_dropout

    def forward(self, attr_idx, edge_idx, n, d):

        hidden = F.relu(self.conv1(attr_idx, edge_idx, n, d))
        hidden = F.dropout(hidden, p=self.p_dropout, training=self.training)
        hidden = self.conv2(hidden, edge_idx)
        return hidden


class GAT(torch.nn.Module):
    def __init__(self, n_features, n_classes, n_hidden, k_heads, p_dropout=0.6):
        super().__init__()
        self.p_dropout = p_dropout
        self.conv1 = SparseGATConv(n_features, n_hidden, heads=k_heads, dropout=self.p_dropout)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(
            n_hidden * k_heads, n_classes, heads=1, concat=True, dropout=self.p_dropout)

    def forward(self, attr_idx, edge_idx, n, d):
        # Regular GAT uses dropout on attributes and adjacency matrix
        x = F.elu(self.conv1(attr_idx, edge_idx, n, d))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv2(x, edge_idx)
        return F.log_softmax(x, dim=1)


class APPNPNet(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden, k_hops, alpha, p_dropout=0.5):
        super().__init__()
        self.lin1 = SparseLinear(n_features, n_hidden, bias=False)
        self.lin2 = nn.Linear(n_hidden, n_classes, bias=False)
        self.prop = APPNP(k_hops, alpha)
        self.p_dropout = p_dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, attr_idx, edge_idx, n, d):
        # Regular PPNP uses dropout on attributes and adjacency matrix
        x = F.relu(self.lin1(attr_idx, torch.ones_like(attr_idx[0]), n))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_idx)
        return F.log_softmax(x, dim=1)


class CNN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        # Normalization for MNIST.
        # Our data is binarized, so we need to do this as part of the model.
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.normalize(x)
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        return x


class GIN(nn.Module):
    def __init__(self, dataset, num_layers, hidden, train_eps=False, mode='cat'):
        super().__init__()
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(dataset.num_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
        ), train_eps=train_eps)
        self.convs = nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden),
                ), train_eps=train_eps))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = nn.Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        x = self.jump(xs)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__
