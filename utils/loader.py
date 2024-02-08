import torch
import random
import numpy as np
from denoising_model import denoising_model
from gaussian_ddpm_losses import gaussian_ddpm_losses
import torch.nn.functional as F
import torch.distributed as dist



def load_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def load_model(params):
    params_ = params.copy()
    model = denoising_model(**params_)
    return model


def load_model_optimizer(params, config_train, device):
    model = load_model(params)
    devices_id = torch.device('cuda', torch.distributed.get_rank())
    model=model.to(torch.device('cuda', torch.distributed.get_rank()))
    model=torch.nn.parallel.DistributedDataParallel(model, device_ids=[devices_id], output_device=devices_id, find_unused_parameters=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config_train.lr, weight_decay=config_train.weight_decay)
    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_train.lr_decay)
    return model, optimizer, scheduler


def load_data(config):
    from utils.data_loader import dataloader
    return dataloader(config)


def load_batch(batch, device):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    x_b = batch.x.to(device_id).view(-1, batch.x.shape[-1]).float()
    adj_b = batch.edge_index.to(device_id).view(2, -1)
    y_b = batch.y.to(device_id)
    line_b = batch.pos.to(device_id)

    y_b = F.one_hot(y_b.view(-1), 2).float()
    return x_b, adj_b, y_b, torch.tensor([0]), line_b


def load_loss_fn(config, device):
    if config.diffusion.method == 'Gaussian':
        return gaussian_ddpm_losses(config.diffusion.step, device=device, time_batch=config.train.time_batch, unweighted_MSE=config.train.unweighted_MSE)


def load_model_params(config):
    config_m = config.model
    nlabel = config.data.nlabel
    block_size = config.train.block_size
    params_ = {'num_nodes': block_size, 'num_linears': config_m.num_linears, 'nhid': config_m.nhid,
                'nfeat': config.data.nfeat, 'cat_mode':config_m.cat_mode, 'skip':config_m.skip,
                'nlabel': nlabel, 'num_layers': config_m.num_layers, 'types': config.diffusion.method}
    return params_
