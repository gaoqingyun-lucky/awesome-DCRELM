import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
from sklearn.metrics import pairwise_distances as pair
from torch import nn

def construct_graph(fname, features, label, method='heat', topk=1):
    num = len(label)
    dist = None

    if method == 'heat':
        dist = -0.5 * pair(features) ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos':
        features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)

    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    f = open(fname, 'w')
    counter = 0
    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                if label[vv] != label[i]:
                    counter += 1
                f.write('{} {}\n'.format(i, vv))
    f.close()
    print('Error Rate: {}'.format(counter / (num * topk)))


def load_graph(k, graph_k_save_path, graph_save_path, data_path):
    if k:
        path = graph_k_save_path
    else:
        path = graph_save_path

    print("Loading path:", path)

    data = np.loadtxt(data_path, dtype=float)
    n, _ = data.shape

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj

def test_load_graph(k, graph_k_save_path, graph_save_path, data_path):
    if k:
        path = graph_k_save_path
    else:
        path = graph_save_path

    # print("Loading path:", path)
    # if data_path.endswith('.txt'):
    #     data = np.loadtxt(data_path, dtype=float)
    # else:
    data = data_path
    n, _ = data.shape

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class LoadDataset(Dataset):

    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))

def normalize_adj(adj, self_loop=True, symmetry=False):
    """
    normalize the adj matrix
    :param adj: input adj matrix
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not
    :return: the normalized adj matrix
    """
    # add the self_loop
    if self_loop:
        adj_tmp = adj + torch.eye(adj.shape[0]).to('cuda')
    else:
        adj_tmp = adj

    # calculate degree matrix and it's inverse matrix
    d = torch.diag(adj_tmp.sum(0))
    d_inv = torch.linalg.inv(d)

    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        sqrt_d_inv = torch.sqrt(d_inv)
        norm_adj = torch.mm(torch.mm(sqrt_d_inv, adj_tmp), adj_tmp)

    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = torch.mm(d_inv, adj_tmp)

    return norm_adj

def ELM(Total_data,n_hidden,active_fun):
    (n_sample,n_feature) = Total_data.shape
    ELM_model = nn.Linear(n_feature, n_hidden)
    b = torch.tensor(Total_data.astype("float32"))
    tempH = ELM_model(b)
    if active_fun == 'sigmod':
        # H = 1. / (1. + torch.exp(-tempH))
        sig = nn.Sigmoid()
        H = sig(tempH)
    elif active_fun == 'sin':
        H = torch.sin(tempH)
    elif active_fun == 'ReLU':
        H = torch.maximum(torch.tensor(0.),tempH)
    return H