import os.path as osp
import pandas as pd
import torch
from torch_geometric.data import (InMemoryDataset, Data, DataLoader)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class ProcessedDataset(InMemoryDataset):
    pass


def precompute_edge_label_and_reverse(dataset: InMemoryDataset):
    data_list = []
    for data in dataset:
        for idx, dat in enumerate(data):
            u, v = dat.edge_index
            yu, yv = dat.y[u], dat.y[v]
            dat.edge_labels = yu * 2 + yv
            data_list.append(dat)

    new_data, new_slices = InMemoryDataset.collate(data_list)
    new_dataset = ProcessedDataset('.')
    new_dataset.data = new_data
    new_dataset.slices = new_slices
    return new_dataset


class CitationDataset(InMemoryDataset):
    def __init__(self, root=None, split='train', transform=None, pre_transform=None, pre_filter=None):
        assert split in ['train', 'val', 'test']
        super(CitationDataset, self).__init__(root, transform, pre_transform, pre_filter)
        saved_data = pd.read_json(open(root))
        data_list = []
        for idx in saved_data.index:
            graph_data = Data(edge_index=torch.tensor(saved_data.loc[idx].edges),
                              x=torch.tensor(saved_data.loc[idx].node_feature),
                              y=torch.tensor(saved_data.loc[idx].node_target),
                              pos=torch.tensor(saved_data.loc[idx].node_lines))
            data_list.append(graph_data)
        self.data = data_list

        num_nodes = 400
        num_edges = 1000

        self.slices = {
            'x': torch.LongTensor([0, num_nodes]),
            'y': torch.LongTensor([0, num_nodes]),
            'gl': torch.LongTensor([0]),
            'edge_index': torch.LongTensor([0, num_edges])
        }


class BatchedCitationDataset(InMemoryDataset):
    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None):
        super(BatchedCitationDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data = torch.load(root)
        num_nodes = self.data.x.size(0)
        num_edges = self.data.edge_index.size(1)
        self.data = Data(edge_index=self.data.edge_index, x=self.data.x, y=self.data.y)

        self.slices = {
            'x': torch.LongTensor([0, num_nodes]), 
            'y': torch.LongTensor([0, num_nodes]),
            'edge_index': torch.LongTensor([0, num_edges]),
            'batch': torch.LongTensor([0, num_edges])
        }


def prepare_dblp(project):
    path_train = osp.join('.', 'data', f'{project}_train.json')
    path_valid = osp.join('.', 'data', f'{project}_valid.json')
    path_test = osp.join('.', 'data', f'{project}_test.json')
    train_dataset = CitationDataset(root=path_train, split='train')
    val_dataset = CitationDataset(root=path_valid, split='val')
    test_dataset = CitationDataset(root=path_test, split='test')
    return train_dataset, val_dataset, test_dataset


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def dataloader(config):
    train_dataset, val_dataset, test_dataset = map(precompute_edge_label_and_reverse, prepare_dblp(config.dataset))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch, shuffle=False, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch, shuffle=False, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=config.train.batch, shuffle=False, sampler=test_sampler)

    return train_loader, val_loader, test_loader

