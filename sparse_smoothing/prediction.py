import torch
import torch.nn.functional as F
from sparse_smoothing.utils import sample_multiple_graphs, binary_perturb
from tqdm.autonotebook import tqdm


def predict_smooth_gnn(attr_idx, edge_idx, sample_config, model, n, d, nc, batch_size=1):
    """

    Parameters
    ----------
    attr_idx: torch.Tensor [2, ?]
        The indices of the non-zero attributes.
    edge_idx: torch.Tensor [2, ?]
        The indices of the edges.
    sample_config: dict
        Configuration specifying the sampling probabilities
    model : torch.nn.Module
        The GNN model.
    n : int
        Number of nodes
    d : int
        Number of features
    nc : int
        Number of classes
    batch_size : int
        Number of graphs to sample per batch

    Returns
    -------
    votes : array-like [n, nc]
        The votes per class for each node.
    """
    n_samples = sample_config.get('n_samples', 1)

    model.eval()
    votes = torch.zeros((n, nc), dtype=torch.long, device=attr_idx.device)
    with torch.no_grad():
        assert n_samples % batch_size == 0
        nbatches = n_samples // batch_size

        for _ in tqdm(range(nbatches)):
            attr_idx_batch, edge_idx_batch = sample_multiple_graphs(
                    attr_idx=attr_idx, edge_idx=edge_idx,
                    sample_config=sample_config, n=n, d=d, nsamples=batch_size)

            predictions = model(attr_idx=attr_idx_batch, edge_idx=edge_idx_batch,
                                n=batch_size * n, d=d).argmax(1)
            preds_onehot = F.one_hot(predictions, int(nc)).reshape(batch_size, n, nc).sum(0)
            votes += preds_onehot
    return votes.cpu().numpy()


def predict_smooth_pytorch(model, dataloader, n_data, n_classes,
                           data_tuple=True, sample_fn=None, sample_config=None):
    device = next(model.parameters()).device

    n_samples = sample_config.get('n_samples', 1)

    model.eval()
    ncorr = 0
    votes = torch.zeros((n_data, n_classes), dtype=torch.long, device=device)
    for ibatch, data in enumerate(dataloader):
        if data_tuple:
            xb, yb = data[0].to(device), data[1].to(device)
        else:
            data.to(device)
            xb = data
            yb = data.y

        batch_idx = ibatch * dataloader.batch_size
        for _ in tqdm(range(n_samples)):
            data_perturbed = sample_fn(xb, sample_config)
            preds = model(data_perturbed).argmax(1)
            preds_onehot = F.one_hot(preds, n_classes)
            votes[batch_idx:batch_idx + yb.shape[0]] += preds_onehot
        ncorr += (votes[batch_idx:batch_idx + yb.shape[0]].argmax(1) == yb).sum().item()
    return votes.cpu().numpy(), ncorr / n_data
