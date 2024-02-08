import pandas as pd
import numpy as np
import torch
import random
from gensim.models import Word2Vec
from my_tokenizer import extract_tokens


def preprocess_features(feature, block_size):
    """Row-normalize feature matrix and convert to tuple representation"""
    max_length = block_size
    feature = np.array(feature)
    pad = max_length - feature.shape[0]  # padding for each epoch
    feature = np.pad(feature, ((0, pad), (0, 0)), mode='constant')

    return feature

def preprocess_edges(edges, num_nodes):
    new_edges = [[], []]
    for i,edge in enumerate(edges):
        if edge[0] < num_nodes and edge[1] < num_nodes:
            new_edges[0].append(edge[0])
            new_edges[1].append(edge[1])

    return new_edges

def get_batch_edges(idx_, idx, all_edges):
    batch_edges = [[], []]
    for edge in all_edges:
        if (edge[0] >= idx_ and edge[0] < idx) and (edge[1] >= idx_ and edge[1] < idx):
            batch_edges[0].append(edge[0]-idx_)
            batch_edges[1].append(edge[1]-idx_)

    return batch_edges


def split_dataset(project, data, model, model_type, source_index, block_size):
    save_data = pd.DataFrame(columns=['filename', 'node_feature', 'node_target', 'edges', 'node_lines'])
    for indexs in source_index:
        print(indexs)

        node_features = []
        filename = data.loc[indexs].filename
        nodes = data.loc[indexs].nodes
        if len(nodes) == 0:
            continue
        edges = data.loc[indexs].edges
        nodes = nodes[:block_size]
        num_nodes = len(nodes)
        edges = preprocess_edges(edges, num_nodes)

        nodes_codes = data.loc[indexs].nodes_codes
        node_labels = data.loc[indexs].nodes_label

        # Word2Vec
        for i, code in enumerate(nodes_codes[:num_nodes]):
            emb_seq = []
            code_tokens = extract_tokens(code)
            node_label = node_labels[i]
            type_emb = torch.tensor(model_type[node_label])
            if len(code_tokens)==0:
                feature_out = torch.tensor(np.zeros(128))
            else:
                for token in code_tokens:
                    try:
                        emb_seq.append(model.wv[token])
                    except:
                        emb_seq.append(np.zeros(128))
                emb_seq = torch.tensor(np.array(emb_seq))
                feature_out = torch.sum(emb_seq, 0)
            feature_out = torch.cat((type_emb, feature_out), 0)
            feature_out = feature_out.cpu().detach().numpy().tolist()
            node_features.append(feature_out)

        node_features = preprocess_features(node_features, block_size)
        node_target = data.loc[indexs].node_target
        node_target = node_target[:num_nodes]
        node_lines = data.loc[indexs].code_lines
        node_lines = node_lines[:num_nodes]

        if len(node_target) < block_size:
            pad_size = block_size-len(node_target)
            node_target.extend([0]*pad_size)
            node_lines.extend([-1]*pad_size)

        series = pd.Series({'filename': filename, 'node_feature': node_features, 'node_target': node_target, 'edges': edges, 'node_lines': node_lines})
        save_data = save_data.append(series, ignore_index=True)

    return save_data

if __name__ == '__main__':

    project = "openssl"
    block_size = 400
    datasource = pd.read_json(open(f'./data/{project}_GrVCs.json'))

    all_tokens = []
    for ind in datasource.index:
        codes = datasource.loc[ind].nodes_codes
        tokens = []
        for code in codes:
            code_tokens = extract_tokens(code)
            tokens.append(code_tokens)
        all_tokens.extend(tokens)

    model = Word2Vec(all_tokens, vector_size=128, window=10, min_count=5, workers=12, epochs=10, sg=1)  # gensim=4.3.1

    token_index = {}
    for ind in datasource.index:
        print(ind)
        types = datasource.loc[ind].nodes_label
        for type in types:
            if type not in token_index:
                token_index[type] = [len(token_index) + 1]

    datasource = datasource.sample(frac=1.0).reset_index(drop=True)
    js_all = datasource.to_dict('records')
    total_num = len(js_all)
    train_num = int(total_num * 0.8)
    valid_num = int(total_num * 0.9)
    total_idx = [i for i in range(total_num)]
    random.shuffle(total_idx)

    train_index = total_idx[:train_num]
    val_index = total_idx[train_num:valid_num]
    test_index = total_idx[valid_num:]
    train_data = split_dataset(project, datasource, model, token_index, train_index, block_size)
    val_data = split_dataset(project, datasource, model, token_index, val_index, block_size)
    test_data = split_dataset(project, datasource, model, token_index, test_index, block_size)

    train_data.to_json(f"./data/{project}_train.json")
    test_data.to_json(f"./data/{project}_test.json")
    val_data.to_json(f"./data/{project}_valid.json")

print('a')


