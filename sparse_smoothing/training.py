import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from functools import partial
from sparse_smoothing.utils import sample_multiple_graphs, accuracy
from sparse_smoothing.prediction import predict_smooth_gnn
import time
import logging


def get_time():
    torch.cuda.synchronize()
    return time.time()


def smooth_logits_gnn(attr_idx, edge_idx, model, sample_config, n, d, nc, batch_size=1, idx_nodes=None):
    n_samples = sample_config.get('n_samples', 1)
    mean_softmax = sample_config['mean_softmax']

    assert n_samples % batch_size == 0
    nbatches = n_samples // batch_size

    arng = torch.arange(n, dtype=torch.long,
                        device=attr_idx.device).repeat(batch_size)
    logits = torch.zeros([n, nc], dtype=torch.float, device=attr_idx.device)

    for _ in range(nbatches):
        attr_idx_batch, edge_idx_batch = sample_multiple_graphs(
            attr_idx=attr_idx, edge_idx=edge_idx,
            sample_config=sample_config, n=n, d=d, nsamples=batch_size)

        logits_batch = model(attr_idx=attr_idx_batch, edge_idx=edge_idx_batch,
                             n=batch_size * n, d=d)
        if mean_softmax:
            logits_batch = F.softmax(logits_batch, dim=1)
        logits = logits + scatter_add(logits_batch, arng, dim=0, dim_size=n)

    # divide by n_samples so we have the mean
    logits = logits / n_samples

    # go back to log space if we were averaging in probability space
    if mean_softmax:
        logits = torch.log(torch.clamp(logits, min=1e-20))

    return logits


def smooth_logits_pytorch(data, model, sample_config, sample_fn):
    n_samples = sample_config.get('n_samples', 1)

    logits = []
    for _ in range(n_samples):
        data_perturbed = sample_fn(data, sample_config)
        logits.append(model(data_perturbed))
    return torch.stack(logits).mean(0)


def train_gnn(model, edge_idx, attr_idx, labels, n, d, nc,
              idx_train, idx_val,
              lr, weight_decay, patience, max_epochs,
              sample_config=None, display_step=50,
              batch_size=1, early_stopping=True):
    trace_val = []
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    if sample_config is not None:
        model_partial = partial(smooth_logits_gnn, model=model, sample_config=sample_config,
                                n=n, d=d, nc=nc, batch_size=batch_size)
    else:
        model_partial = partial(model, n=n, d=d)

    best_loss = np.inf
    last_time = get_time()

    for it in range(max_epochs):
        logits = model_partial(attr_idx=attr_idx, edge_idx=edge_idx)
        
        loss_train = F.cross_entropy(logits[idx_train], labels[idx_train])
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        loss_val = F.cross_entropy(logits[idx_val], labels[idx_val])
        trace_val.append(loss_val.item())

        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = it
            best_state = {key: value.cpu()
                          for key, value in model.state_dict().items()}
        else:
            if it >= best_epoch + patience and early_stopping:
                break

        if it % display_step == 0:
            acc_train = accuracy(labels, logits, idx_train)
            acc_val = accuracy(labels, logits, idx_val)

            current_time = get_time()
            logging.info(f'Epoch {it:4}: loss_train: {loss_train.item():.5f}, loss_val: {loss_val.item():.5f} '
                         f'acc_train: {acc_train:.5f}, acc_val: {acc_val:.5f} ({current_time - last_time:.3f}s)')
            last_time = current_time
    
    print('best_epoch', best_epoch)
    model.load_state_dict(best_state)
    return trace_val


def run_epoch_pytorch(
        model, optimizer, dataloader, nsamples, train, data_tuple=True, device='cuda'):
    """
    Run one epoch of training or evaluation.

    Args:
        model: The model used for prediction
        optimizer: Optimization algorithm for the model
        dataloader: Dataloader providing the data to run our model on
        nsamples: Number of samples over which the dataloader iterates
        train: Whether this epoch is used for training or evaluation
        data_tuple: Whether dataloader returns a tuple (x, y)
        device: Target device for computation

    Returns:
        Loss and accuracy in this epoch.
    """
    start = get_time()

    epoch_loss = 0.0
    epoch_acc = 0.0

    # Iterate over data
    for data in dataloader:
        if data_tuple:
            xb, yb = data[0].to(device), data[1].to(device)
        else:
            data.to(device)
            xb = data
            yb = data.y

        # zero the parameter gradients
        if train:
            optimizer.zero_grad()

        # forward
        with torch.set_grad_enabled(train):
            pred = model(xb)
            loss = F.cross_entropy(pred, yb)
            top1 = torch.argmax(pred, dim=1)
            ncorrect = torch.sum(top1 == yb)

            # backward + optimize only if in training phase
            if train:
                loss.backward()
                optimizer.step()

        # statistics
        epoch_loss += loss.item()
        epoch_acc += ncorrect.item()

    epoch_loss /= nsamples
    epoch_acc /= nsamples
    epoch_time = get_time() - start
    return epoch_loss, epoch_acc, epoch_time


def train_pytorch(
        model, dataloaders, optimizer, lr_scheduler, n_samples,
        lr, weight_decay, patience, max_epochs, data_tuple=True,
        sample_fn=None, sample_config=None):

    if sample_config is not None:
        assert sample_fn is not None
        model_partial = partial(smooth_logits_pytorch,
                                model=model, sample_config=sample_config,
                                sample_fn=sample_fn)
    else:
        model_partial = partial(model)

    trace_val = []
    best_loss = np.inf
    for epoch in range(max_epochs):
        model.train()
        train_loss, train_acc, train_time = run_epoch_pytorch(
            model_partial, optimizer, dataloaders['train'], n_samples['train'],
            train=True, data_tuple=data_tuple)
        logging.info(f"Epoch {epoch + 1: >3}/{max_epochs}, "
                     f"train loss: {train_loss:.2e}, "
                     f"accuracy: {train_acc * 100:.2f}% ({train_time:.2f}s)")

        model.eval()
        val_loss, val_acc, val_time = run_epoch_pytorch(
            model_partial, None, dataloaders['val'], n_samples['val'],
            train=False, data_tuple=data_tuple)
        trace_val.append(val_loss)
        logging.info(f"Epoch {epoch + 1: >3}/{max_epochs}, "
                     f"val loss: {val_loss:.2e}, "
                     f"accuracy: {val_acc * 100:.2f}% ({val_time:.2f}s)")

        lr_scheduler.step()

        if val_loss < best_loss:
            best_epoch = epoch
            best_loss = val_loss
            best_state = {key: value.cpu()
                          for key, value in model.state_dict().items()}

        # Early stopping
        if epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    return trace_val
