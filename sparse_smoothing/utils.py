from typing import List, Union
import torch
import numpy as np
from sparse_smoothing.sparsegraph import SparseGraph
from torch_sparse import coalesce
from torch_geometric.data import Data, Batch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset



def binary_perturb(data, pf_minus, pf_plus):
    """
    Randomly flip bits.

    Parameters
    ----------
    data: torch.Tensor [b, ?, ?]
        The indices of the non-zero elements
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one

    Returns
    -------
    data_perturbed: torch.Tensor [b, ?, ?]
        The indices of the non-zero elements after perturbation
    """

    to_del = torch.cuda.BoolTensor(data.shape).bernoulli_(1 - pf_minus)
    to_add = torch.cuda.BoolTensor(data.shape).bernoulli_(pf_plus)

    data_perturbed = data * to_del + (1 - data) * to_add
    return data_perturbed


def retain_k_elements(data_idx, k, undirected, shape=None):
    """
    Randomly retain k (non-zero) elements.

    Parameters
    ----------
    data_idx: torch.Tensor [2, ?]
        The indices of the non-zero elements.
    k : int
        Number of elements to remove.
    undirected : bool
        If true for every (i, j) also perturb (j, i).
    shape: (int, int)
        If shape=None only retain k non-zero elements,
        else retain k of all possible shape[0]*shape[0] pairs (including zeros).

    Returns
    -------
    per_data_idx: torch.Tensor [2, ?]
        The indices of the non-zero elements after perturbation.
    """
    if undirected:
        data_idx = data_idx[:, data_idx[0] < data_idx[1]]

    if shape is not None:
        n, m = shape
        if undirected:
            # undirected makes sense only for square (adjacency matrices)
            assert n == m 
            total_pairs = n*(n+1)//2
        else:
            total_pairs = n*m

        k = k*data_idx.shape[1]//total_pairs

    rnd_idx = torch.randperm(data_idx.shape[1], device='cuda')[:k]
    per_data_idx = data_idx[:, rnd_idx]

    if undirected:
        per_data_idx = torch.cat((per_data_idx, per_data_idx[[1, 0]]), 1)

    return per_data_idx


def sparse_perturb(data_idx, pf_minus, pf_plus, n, m, undirected):
    """
    Randomly flip bits.

    Parameters
    ----------
    data_idx: torch.Tensor [2, ?]
        The indices of the non-zero elements
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one
    n : int
        The shape of the tensor
    m : int
        The shape of the tensor
    undirected : bool
        If true for every (i, j) also perturb (j, i)

    Returns
    -------
    perturbed_data_idx: torch.Tensor [2, ?]
        The indices of the non-zero elements after perturbation
    """
    if undirected:
        # select only one direction of the edges, ignore self loops
        data_idx = data_idx[:, data_idx[0] < data_idx[1]]

    w_existing = torch.ones_like(data_idx[0])
    to_del = torch.cuda.BoolTensor(data_idx.shape[1]).bernoulli_(pf_minus)
    w_existing[to_del] = 0

    nadd = np.random.binomial(n * m, pf_plus)  # 6x faster than PyTorch
    nadd_with_repl = int(np.log(1 - nadd / (n * m)) / np.log(1 - 1 / (n * m)))
    to_add = data_idx.new_empty([2, nadd_with_repl])
    to_add[0].random_(n * m)
    to_add[1] = to_add[0] % m
    to_add[0] = to_add[0] // m
    if undirected:
        # select only one direction of the edges, ignore self loops
        assert n == m
        to_add = to_add[:, to_add[0] < to_add[1]]

    w_added = torch.ones_like(to_add[0])

    # if an edge already exists but has been removed do not add it back
    # hence we coalesce with the min value
    joined, weights = coalesce(torch.cat((data_idx, to_add), 1),
                               torch.cat((w_existing, w_added), 0),
                               n, m, 'min')

    per_data_idx = joined[:, weights > 0]

    if undirected:
        per_data_idx = torch.cat((per_data_idx, per_data_idx[[1, 0]]), 1)

    return per_data_idx


def sparse_perturb_adj_batch(data_idx, nnodes, pf_minus, pf_plus, undirected):
    """
    Randomly flip bits.

    Parameters
    ----------
    data_idx: torch.Tensor [2, ?]
        The indices of the non-zero elements
    nnodes : array_like, dtype=int
        Number of nodes per graph
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one
    undirected : bool
        If true for every (i, j) also perturb (j, i)

    Returns
    -------
    perturbed_data_idx: torch.Tensor [2, ?]
        The indices of the non-zero elements after perturbation
    """
    if undirected:
        # select only one direction of the edges, ignore self loops
        data_idx = data_idx[:, data_idx[0] < data_idx[1]]

    w_existing = torch.ones_like(data_idx[0])
    to_del = torch.cuda.BoolTensor(data_idx.shape[1]).bernoulli_(pf_minus)
    w_existing[to_del] = 0

    offsets = torch.cat((nnodes.new_zeros(1), torch.cumsum(nnodes, dim=0)[:-1]))
    nedges = torch.cumsum(nnodes**2, dim=0)
    offsets2 = torch.cat((nedges.new_zeros(1), nedges[:-1]))
    nedges_total = nedges[-1].item()
    nadd = np.random.binomial(nedges_total, pf_plus)  # 6x faster than PyTorch
    nadd_with_repl = int(np.log(1 - nadd / nedges_total) / np.log(1 - 1 / nedges_total))
    to_add = data_idx.new_empty([2, nadd_with_repl])
    to_add[0].random_(nedges_total)
    add_batch = (to_add[0][:, None] >= nedges[None, :]).sum(1)
    to_add[0] -= offsets2[add_batch]
    to_add[1] = to_add[0] % nnodes[add_batch]
    to_add[0] = to_add[0] // nnodes[add_batch]
    to_add += offsets[add_batch][None, :]
    if undirected:
        # select only one direction of the edges, ignore self loops
        to_add = to_add[:, to_add[0] < to_add[1]]

    w_added = torch.ones_like(to_add[0])

    # if an edge already exists but has been removed do not add it back
    # hence we coalesce with the min value
    nnodes_total = torch.sum(nnodes)
    joined, weights = coalesce(torch.cat((data_idx, to_add), 1),
                               torch.cat((w_existing, w_added), 0),
                               nnodes_total, nnodes_total, 'min')

    per_data_idx = joined[:, weights > 0]

    if undirected:
        per_data_idx = torch.cat((per_data_idx, per_data_idx[[1, 0]]), 1)

    # Check that there are no off-diagonal edges
    # batch0 = (to_add[0][:, None] >= nnodes.cumsum(0)[None, :]).sum(1)
    # batch1 = (to_add[1][:, None] >= nnodes.cumsum(0)[None, :]).sum(1)
    # assert torch.all(batch0 == batch1)

    return per_data_idx


def sparse_perturb_multiple(data_idx, pf_minus, pf_plus, n, m, undirected, nsamples, offset_both_idx):
    """
    Randomly flip bits.

    Parameters
    ----------
    data_idx: torch.Tensor [2, ?]
        The indices of the non-zero elements
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one
    n : int
        The shape of the tensor
    m : int
        The shape of the tensor
    undirected : bool
        If true for every (i, j) also perturb (j, i)
    nsamples : int
        Number of perturbed samples
    offset_both_idx : bool
        Whether to offset both matrix indices (for adjacency matrix)

    Returns
    -------
    perturbed_data_idx: torch.Tensor [2, ?]
        The indices of the non-zero elements of multiple concatenated matrices
        after perturbation
    """
    if undirected:
        # select only one direction of the edges, ignore self loops
        data_idx = data_idx[:, data_idx[0] < data_idx[1]]

    idx_copies = copy_idx(data_idx, n, nsamples, offset_both_idx)
    w_existing = torch.ones_like(idx_copies[0])
    to_del = torch.cuda.BoolTensor(idx_copies.shape[1]).bernoulli_(pf_minus)
    w_existing[to_del] = 0

    if offset_both_idx:
        assert n == m
        nadd_persample_np = np.random.binomial(n * m, pf_plus, size=nsamples)  # 6x faster than PyTorch
        nadd_persample = torch.cuda.FloatTensor(nadd_persample_np)
        nadd_persample_with_repl = torch.round(torch.log(1 - nadd_persample / (n * m))
                                               / np.log(1 - 1 / (n * m))).long()
        nadd_with_repl = nadd_persample_with_repl.sum()
        to_add = data_idx.new_empty([2, nadd_with_repl])
        to_add[0].random_(n * m)
        to_add[1] = to_add[0] % m
        to_add[0] = to_add[0] // m
        to_add = offset_idx(to_add, nadd_persample_with_repl, m, [0, 1])
        if undirected:
            # select only one direction of the edges, ignore self loops
            to_add = to_add[:, to_add[0] < to_add[1]]
    else:
        nadd = np.random.binomial(nsamples * n * m, pf_plus)  # 6x faster than PyTorch
        nadd_with_repl = int(np.round(np.log(1 - nadd / (nsamples * n * m))
                                      / np.log(1 - 1 / (nsamples * n * m))))
        to_add = data_idx.new_empty([2, nadd_with_repl])
        to_add[0].random_(nsamples * n * m)
        to_add[1] = to_add[0] % m
        to_add[0] = to_add[0] // m

    w_added = torch.ones_like(to_add[0])

    if offset_both_idx:
        mb = nsamples * m
    else:
        mb = m

    # if an edge already exists but has been removed do not add it back
    # hence we coalesce with the min value
    joined, weights = coalesce(torch.cat((idx_copies, to_add), 1),
                               torch.cat((w_existing, w_added), 0),
                               nsamples * n, mb, 'min')

    per_data_idx = joined[:, weights > 0]

    if undirected:
        per_data_idx = torch.cat((per_data_idx, per_data_idx[[1, 0]]), 1)

    # Check that there are no off-diagonal edges
    # if offset_both_idx:
    #     batch0 = to_add[0] // n
    #     batch1 = to_add[1] // n
    #     assert torch.all(batch0 == batch1)

    return per_data_idx


def to_undirected(edge_idx, n):
    """
    Keep only edges that appear in both directions.

    Parameters
    ----------
    edge_idx : torch.Tensor [2, ?]
        The indices of the edges
    n : int
        Number of nodes

    Returns
    -------
    edge_idx : torch.Tensor [2, ?]
        The indices of the edges that appear in both directions
    """
    joined = torch.cat((edge_idx, edge_idx[[1, 0]]), 1)
    edge_idx, value = coalesce(joined, torch.ones_like(joined[0]), n, n, 'add')

    # keep only the edges that appear twice
    edge_idx = edge_idx[:, value > 1]

    return edge_idx


def accuracy(labels, logits, idx):
    return (labels[idx] == logits[idx].argmax(1)).sum().item() / len(idx)


def accuracy_majority(labels, votes, idx):
    return (votes.argmax(1)[idx] == labels[idx]).mean()


def split(labels, n_per_class=20, seed=0):
    """
    Randomly split the training data.

    Parameters
    ----------
    labels: array-like [n_nodes]
        The class labels
    n_per_class : int
        Number of samples per class
    seed: int
        Seed

    Returns
    -------
    split_train: array-like [n_per_class * nc]
        The indices of the training nodes
    split_val: array-like [n_per_class * nc]
        The indices of the validation nodes
    split_test array-like [n_nodes - 2*n_per_class * nc]
        The indices of the test nodes
    """
    np.random.seed(seed)
    nc = labels.max() + 1

    split_train, split_val = [], []
    for l in range(nc):
        perm = np.random.permutation((labels == l).nonzero()[0])
        split_train.append(perm[:n_per_class])
        split_val.append(perm[n_per_class:2 * n_per_class])

    split_train = np.random.permutation(np.concatenate(split_train))
    split_val = np.random.permutation(np.concatenate(split_val))

    assert split_train.shape[0] == split_val.shape[0] == n_per_class * nc

    split_test = np.setdiff1d(np.arange(len(labels)), np.concatenate((split_train, split_val)))

    return split_train, split_val, split_test


def load_and_standardize(file_name):
    """
    Run gust.standardize() + make the attributes binary.

    Parameters
    ----------
    file_name
        Name of the file to load.
    Returns
    -------
    graph: gust.SparseGraph
        The standardized graph

    """
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        if 'type' in loader:
            del loader['type']
        graph = SparseGraph.from_flat_dict(loader)
    
    graph.standardize()
    
    # binarize
    graph._flag_writeable(True)
    graph.adj_matrix[graph.adj_matrix != 0] = 1
    graph.attr_matrix[graph.attr_matrix != 0] = 1
    graph._flag_writeable(False)

    return graph
    

def sample_perturbed_mnist(data, sample_config):
    pf_minus = sample_config.get('pf_minus_att', 0)
    pf_plus = sample_config.get('pf_plus_att', 0)
    return binary_perturb(data, pf_minus, pf_plus)


def sample_one_graph(attr_idx, edge_idx, sample_config, n, d):
    """
    Perturb the structure and node attributes.

    Parameters
    ----------
    attr_idx: torch.Tensor [2, ?]
        The indices of the non-zero attributes.
    edge_idx: torch.Tensor [2, ?]
        The indices of the edges.
    sample_config: dict
        Configuration specifying the sampling probabilities
    n : int
        Number of nodes
    d : int
        Number of features

    Returns
    -------
    attr_idx: torch.Tensor [2, ?]
        The indices of the non-zero attributes after perturbation.
    edge_idx: torch.Tensor [2, ?]
        The indices of the edges after perturbation.
    """
    pf_plus_adj = sample_config.get('pf_plus_adj', 0)
    pf_plus_att = sample_config.get('pf_plus_att', 0)

    pf_minus_adj = sample_config.get('pf_minus_adj', 0)
    pf_minus_att = sample_config.get('pf_minus_att', 0)

    per_attr_idx = sparse_perturb(data_idx=attr_idx, n=n, m=d, undirected=False,
                                  pf_minus=pf_minus_att, pf_plus=pf_plus_att)

    per_edge_idx = sparse_perturb(data_idx=edge_idx, n=n, m=n, undirected=True,
                                  pf_minus=pf_minus_adj, pf_plus=pf_plus_adj)

    return per_attr_idx, per_edge_idx


def sample_batch_pyg(data, sample_config):
    """
    Perturb the structure and node attributes.

    Parameters
    ----------
    data: torch_geometric.data.Batch
        Dataset containing the attributes, edge indices, and batch-ID
    sample_config: dict
        Configuration specifying the sampling probabilities

    Returns
    -------
    per_data: torch_geometric.Dataset
        Dataset containing the perturbed graphs
    """
    pf_plus_adj = sample_config.get('pf_plus_adj', 0)
    pf_plus_att = sample_config.get('pf_plus_att', 0)

    pf_minus_adj = sample_config.get('pf_minus_adj', 0)
    pf_minus_att = sample_config.get('pf_minus_att', 0)

    per_x = binary_perturb(data.x, pf_minus_att, pf_plus_att)

    per_edge_index = sparse_perturb_adj_batch(
            data_idx=data.edge_index, nnodes=torch.bincount(data.batch),
            pf_minus=pf_minus_adj, pf_plus=pf_plus_adj,
            undirected=True)

    per_data = Batch(batch=data.batch, x=per_x, edge_index=per_edge_index)

    return per_data


def sample_multiple_graphs(attr_idx, edge_idx, sample_config, n, d, nsamples):
    """
    Perturb the structure and node attributes.

    Parameters
    ----------
    attr_idx: torch.Tensor [2, ?]
        The indices of the non-zero attributes.
    edge_idx: torch.Tensor [2, ?]
        The indices of the edges.
    sample_config: dict
        Configuration specifying the sampling probabilities
    n : int
        Number of nodes
    d : int
        Number of features
    nsamples : int
        Number of samples

    Returns
    -------
    attr_idx: torch.Tensor [2, ?]
        The indices of the non-zero attributes after perturbation.
    edge_idx: torch.Tensor [2, ?]
        The indices of the edges after perturbation.
    """
    pf_plus_adj = sample_config.get('pf_plus_adj', 0)
    pf_plus_att = sample_config.get('pf_plus_att', 0)

    pf_minus_adj = sample_config.get('pf_minus_adj', 0)
    pf_minus_att = sample_config.get('pf_minus_att', 0)

    if pf_minus_att + pf_plus_att > 0:
        per_attr_idx = sparse_perturb_multiple(data_idx=attr_idx, n=n, m=d, undirected=False,
                                               pf_minus=pf_minus_att, pf_plus=pf_plus_att,
                                               nsamples=nsamples, offset_both_idx=False)
    else:
        per_attr_idx = copy_idx(idx=attr_idx, dim_size=n, ncopies=nsamples, offset_both_idx=False)

    if pf_minus_adj + pf_plus_adj > 0:
        per_edge_idx = sparse_perturb_multiple(data_idx=edge_idx, n=n, m=n, undirected=True,
                                               pf_minus=pf_minus_adj, pf_plus=pf_plus_adj,
                                               nsamples=nsamples, offset_both_idx=True)
    else:
        per_edge_idx = copy_idx(idx=edge_idx, dim_size=n, ncopies=nsamples, offset_both_idx=True)

    return per_attr_idx, per_edge_idx


def collate(attr_idx_list: List[torch.LongTensor],
            edge_idx_list: List[torch.LongTensor], n: int, d: int):
    attr_idx = torch.cat(attr_idx_list, dim=1)
    edge_idx = torch.cat(edge_idx_list, dim=1)

    attr_lens = attr_idx.new_tensor([idx.shape[1] for idx in attr_idx_list])
    edge_lens = edge_idx.new_tensor([idx.shape[1] for idx in edge_idx_list])
    attr_idx = offset_idx(attr_idx, attr_lens, n, [0])
    edge_idx = offset_idx(edge_idx, edge_lens, n, [0, 1])

    return attr_idx, edge_idx


def copy_idx(idx: torch.LongTensor, dim_size: int, ncopies: int, offset_both_idx: bool):
    idx_copies = idx.repeat(1, ncopies)

    offset = dim_size * torch.arange(ncopies, dtype=torch.long,
                                     device=idx.device)[:, None].expand(ncopies, idx.shape[1]).flatten()

    if offset_both_idx:
        idx_copies += offset[None, :]
    else:
        idx_copies[0] += offset

    return idx_copies


def offset_idx(idx_mat: torch.LongTensor, lens: torch.LongTensor, dim_size: int, indices: List[int] = [0]):
    offset = dim_size * torch.arange(len(lens), dtype=torch.long,
                                     device=idx_mat.device).repeat_interleave(lens, dim=0)

    idx_mat[indices, :] += offset[None, :]
    return idx_mat


def get_mnist_dataloaders(batch_size, random_seed=0, num_workers=-1, pin_memory=False, root='../dataset_cache', shuffle=True):
    dataset_dev = datasets.MNIST(
        root=root, train=True, download=True, transform=transforms.ToTensor())
    dataset_test = datasets.MNIST(
        root=root, train=False, download=True, transform=transforms.ToTensor())

    x_dev_bin = (dataset_dev.data > 0.5).float()
    x_test_bin = (dataset_test.data > 0.5).float()
   
    indices = np.arange(len(dataset_dev))
    nvalid = 5000

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[nvalid:], indices[:nvalid]


    dataset_train_bin = TensorDataset(x_dev_bin[train_idx], dataset_dev.targets[train_idx])
    dataset_val_bin = TensorDataset(x_dev_bin[valid_idx], dataset_dev.targets[valid_idx])
    dataset_test_bin = TensorDataset(x_test_bin, dataset_test.targets)

    n_images = {'train': len(train_idx),
                'val': len(valid_idx),
                'test': len(dataset_test)}

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        dataset_train_bin, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory)
    dataloaders['val'] = torch.utils.data.DataLoader(
        dataset_val_bin, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory)

    dataloaders['test'] = torch.utils.data.DataLoader(
        dataset_test_bin, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return dataloaders, n_images
