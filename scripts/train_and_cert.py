from sacred import Experiment
import seml

ex = Experiment()
seml.setup_logger(ex)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(
            db_collection, overwrite=overwrite))

    # default params
    dataset = 'data/cora_ml.npz'
    n_per_class = 20
    seed = 42

    patience = 50
    max_epochs = 3000
    lr = 1e-3
    weight_decay = 1e-3

    model = 'GCN'
    n_hidden = 64
    p_dropout = 0.5

    pf_plus_adj = 0.0
    pf_minus_adj = 0.0

    pf_plus_att = 0.01
    pf_minus_att = 0.6

    n_samples_train = 1
    batch_size_train = 1

    n_samples_pre_eval = 10
    n_samples_eval = 1000
    batch_size_eval = 10

    mean_softmax = False
    conf_alpha = 0.01
    early_stopping = True

    save_dir = 'temp_dir'


@ex.automain
def run(_config, dataset, n_per_class, seed,
        patience, max_epochs, lr, weight_decay, model, n_hidden, p_dropout,
        pf_plus_adj, pf_plus_att, pf_minus_adj, pf_minus_att, conf_alpha,
        n_samples_train, n_samples_pre_eval, n_samples_eval, mean_softmax, early_stopping,
        batch_size_train, batch_size_eval, save_dir,
        ):
    import numpy as np
    import torch
    from sparse_smoothing.models import GCN, GAT, APPNPNet, CNN_MNIST, GIN
    from sparse_smoothing.training import train_gnn, train_pytorch
    from sparse_smoothing.prediction import predict_smooth_gnn, predict_smooth_pytorch
    from sparse_smoothing.cert import binary_certificate, joint_binary_certificate
    from sparse_smoothing.utils import (load_and_standardize, split, accuracy_majority,
                                        sample_perturbed_mnist, sample_batch_pyg, get_mnist_dataloaders)
    from torch_geometric.datasets import TUDataset
    from torch_geometric.data import DataLoader as PyGDataLoader
    print(_config)

    if dataset.lower() not in ['mnist', 'mutag', 'proteins']:
        batch_size_train = min(batch_size_train, n_samples_train)
        batch_size_eval = min(batch_size_eval, n_samples_eval)

    sample_config = {
        'n_samples': n_samples_train,
        'pf_plus_adj': pf_plus_adj,
        'pf_plus_att': pf_plus_att,
        'pf_minus_adj': pf_minus_adj,
        'pf_minus_att': pf_minus_att,
    }

    # if we need to sample at least once and at least one flip probability is non-zero
    if n_samples_train > 0 and (pf_plus_adj+pf_plus_att+pf_minus_adj+pf_minus_att > 0):
        sample_config_train = sample_config
        sample_config_train['mean_softmax'] = mean_softmax
    else:
        sample_config_train = None
    sample_config_eval = sample_config.copy()
    sample_config_eval['n_samples'] = n_samples_eval
    
    sample_config_pre_eval = sample_config.copy()
    sample_config_pre_eval['n_samples'] = n_samples_pre_eval

    if dataset.lower() == 'mnist':
        assert pf_plus_adj == pf_minus_adj == 0

        dataloaders, n_images = get_mnist_dataloaders(
            batch_size_train, random_seed=seed, num_workers=0, pin_memory=True, shuffle=True)

        model = CNN_MNIST().cuda()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1)
        trace_val = train_pytorch(
            model, dataloaders, optimizer, lr_scheduler, n_images,
            lr, weight_decay, patience, max_epochs,
            data_tuple=True, sample_fn=sample_perturbed_mnist,
            sample_config=sample_config_train)

        # reset the dataloaders for evaluation
        dataloaders, n_images = get_mnist_dataloaders(
            batch_size_eval, random_seed=seed, num_workers=0, pin_memory=True, shuffle=False)

        votes_dict = {}
        pre_votes_dict = {}
        acc_majority = {}
        idx = {}
        for split_name in ['train', 'val', 'test']:
            pre_votes_dict[split_name], _ = predict_smooth_pytorch(
                model, dataloaders[split_name], n_images[split_name], n_classes=10,
                data_tuple=True, sample_fn=sample_perturbed_mnist,
                sample_config=sample_config_pre_eval)

        for split_name in ['train', 'val', 'test']:
            votes_dict[split_name], acc_majority[split_name] = predict_smooth_pytorch(
                model, dataloaders[split_name], n_images[split_name], n_classes=10,
                data_tuple=True, sample_fn=sample_perturbed_mnist,
                sample_config=sample_config_eval)
            idx[split_name] = np.arange(votes_dict[split_name].shape[0])

        idx['val'] += idx['train'][-1] + 1
        idx['test'] += idx['val'][-1] + 1

        pre_votes = np.concatenate((pre_votes_dict['train'], pre_votes_dict['val'], pre_votes_dict['test']), axis=0)
        votes = np.concatenate((votes_dict['train'], votes_dict['val'], votes_dict['test']), axis=0)
        votes_max = votes_dict['test'].max(1)
    elif dataset.lower() in ['mutag', 'proteins']:
        pyg_dataset = TUDataset(
            f'../dataset_cache/{dataset.lower()}', dataset.upper())
        pyg_dataset.data.edge_attr = None
        # Caution: Degrees as features if pyg_dataset.x is None.

        n_graphs = {'train': int(0.8 * len(pyg_dataset)),
                    'val': int(0.1 * len(pyg_dataset))}
        n_graphs['test'] = len(pyg_dataset) - \
            n_graphs['train'] - n_graphs['val']

        dataloaders = {}
        dataloaders['train'] = PyGDataLoader(
            pyg_dataset[:n_graphs['train']], batch_size_train, shuffle=True)
        dataloaders['val'] = PyGDataLoader(pyg_dataset[n_graphs['train']:n_graphs['train'] + n_graphs['val']],
                                           batch_size_train, shuffle=False)
        dataloaders['test'] = PyGDataLoader(pyg_dataset[n_graphs['train'] + n_graphs['val']:],
                                            batch_size_train, shuffle=False)

        model = GIN(pyg_dataset, 2, n_hidden).cuda()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.5)
        trace_val = train_pytorch(
            model, dataloaders, optimizer, lr_scheduler, n_graphs,
            lr, weight_decay, patience, max_epochs,
            data_tuple=False, sample_fn=sample_batch_pyg,
            sample_config=sample_config_train)

        votes_dict = {}
        acc_majority = {}
        idx = {}
        for split_name in ['train', 'val', 'test']:
            votes_dict[split_name], acc_majority[split_name] = predict_smooth_pytorch(
                model, dataloaders[split_name], n_graphs[split_name],
                n_classes=pyg_dataset.num_classes,
                data_tuple=False, sample_fn=sample_batch_pyg,
                sample_config=sample_config_eval)
            idx[split_name] = np.arange(votes_dict[split_name].shape[0])

        idx['val'] += idx['train'][-1] + 1
        idx['test'] += idx['val'][-1] + 1

        votes = np.concatenate(
            (votes_dict['train'], votes_dict['val'], votes_dict['test']), axis=0)
        votes_max = votes_dict['test'].max(1)
    else:
        graph = load_and_standardize(dataset)

        edge_idx = torch.LongTensor(
            np.stack(graph.adj_matrix.nonzero())).cuda()
        attr_idx = torch.LongTensor(
            np.stack(graph.attr_matrix.nonzero())).cuda()
        labels = torch.LongTensor(graph.labels).cuda()

        n, d = graph.attr_matrix.shape
        nc = graph.labels.max() + 1

        idx = {}
        idx['train'], idx['val'], idx['test'] = split(
            labels=graph.labels, n_per_class=n_per_class, seed=seed)

        if model.lower() == 'gcn':
            model = GCN(n_features=d, n_classes=nc, n_hidden=n_hidden, p_dropout=p_dropout).cuda()
        elif model.lower() == 'gat':
            # divide the number of hidden units by the number of heads
            #  to match the overall number of paramters
            model = GAT(n_features=d, n_classes=nc, n_hidden=n_hidden // 8,
                        k_heads=8, p_dropout=p_dropout).cuda()
        elif model.lower() == 'appnp':
            model = APPNPNet(n_features=d, n_classes=nc, n_hidden=n_hidden,
                             k_hops=10, alpha=0.15, p_dropout=p_dropout).cuda()
        else:
            raise ValueError(f"Model {model} not implemented.")

        trace_val = train_gnn(model=model, edge_idx=edge_idx, attr_idx=attr_idx, labels=labels, n=n, d=d, nc=nc,
                              idx_train=idx['train'], idx_val=idx['val'], lr=lr, weight_decay=weight_decay,
                              patience=patience, max_epochs=max_epochs, display_step=10,
                              sample_config=sample_config_train,
                              batch_size=batch_size_train, early_stopping=early_stopping)

        pre_votes = predict_smooth_gnn(attr_idx=attr_idx, edge_idx=edge_idx,
                                   sample_config=sample_config_pre_eval,
                                   model=model, n=n, d=d, nc=nc,
                                   batch_size=batch_size_eval)

        votes = predict_smooth_gnn(attr_idx=attr_idx, edge_idx=edge_idx,
                                   sample_config=sample_config_eval,
                                   model=model, n=n, d=d, nc=nc,
                                   batch_size=batch_size_eval)

        acc_majority = {}
        for split_name in ['train', 'val', 'test']:
            acc_majority[split_name] = accuracy_majority(
                votes=votes, labels=graph.labels, idx=idx[split_name])

        votes_max = votes.max(1)[idx['test']]

    agreement = (votes.argmax(1) == pre_votes.argmax(1)).mean() 

    # we are perturbing ONLY the ATTRIBUTES
    if pf_plus_adj == 0 and pf_minus_adj == 0:
        print('Just ATT')
        grid_base, grid_lower, grid_upper = binary_certificate(
            votes=votes, pre_votes=pre_votes, n_samples=n_samples_eval, conf_alpha=conf_alpha,
            pf_plus=pf_plus_att, pf_minus=pf_minus_att)
    # we are perturbing ONLY the GRAPH
    elif pf_plus_att == 0 and pf_minus_att == 0:
        print('Just ADJ')
        grid_base, grid_lower, grid_upper = binary_certificate(
            votes=votes, pre_votes=pre_votes, n_samples=n_samples_eval, conf_alpha=conf_alpha,
            pf_plus=pf_plus_adj, pf_minus=pf_minus_adj)
    else:
        grid_base, grid_lower, grid_upper = joint_binary_certificate(
            votes=votes, pre_votes=pre_votes, n_samples=n_samples_eval, conf_alpha=conf_alpha,
            pf_plus_adj=pf_plus_adj, pf_minus_adj=pf_minus_adj,
            pf_plus_att=pf_plus_att, pf_minus_att=pf_minus_att)

    mean_max_ra_base = (grid_base > 0.5)[:, :, 0].argmin(1).mean()
    mean_max_rd_base = (grid_base > 0.5)[:, 0, :].argmin(1).mean()
    mean_max_ra_loup = (grid_lower >= grid_upper)[:, :, 0].argmin(1).mean()
    mean_max_rd_loup = (grid_lower >= grid_upper)[:, 0, :].argmin(1).mean()


    run_id = _config['overwrite']
    db_collection = _config['db_collection']

    dict_to_save = {'idx_train': idx['train'],
                    'idx_val': idx['val'],
                    'idx_test': idx['test'],
                    'state_dict': model.state_dict(),
                    'trace_val': np.array(trace_val),
                    'pre_votes': pre_votes,
                    'votes': votes,
                    'grid_base': grid_base,
                    'grid_lower': grid_lower,
                    'grid_upper': grid_upper, 
                    }
    torch.save(dict_to_save,
               f'{save_dir}/{db_collection}_{run_id}')

    # the returned result will be written into the database
    results = {
        'acc_majority_train': acc_majority['train'],
        'acc_majority_val': acc_majority['val'],
        'acc_majority_test': acc_majority['test'],
        'above_99': (votes_max >= 0.99 * n_samples_eval).mean(),
        'above_95': (votes_max >= 0.95 * n_samples_eval).mean(),
        'above_90': (votes_max >= 0.90 * n_samples_eval).mean(),
        'mean_max_ra_base': mean_max_ra_base,
        'mean_max_rd_base': mean_max_rd_base,
        'mean_max_ra_loup': mean_max_ra_loup,
        'mean_max_rd_loup': mean_max_rd_loup, 
        'agreement': agreement,
    }

    return results
