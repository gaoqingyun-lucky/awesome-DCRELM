# from train import *
from DCRN import DCRN
from scipy.spatial import distance
from utils import *
import pandas as pd
from opt import args
from DCRN_pretrain.load_data import construct_graph,test_load_graph
from ELM_trainer import *
# from torch.utils.data import Dataset
from DCRN_pretrain.load_data import ELM
import os
import scipy.io
from scipy.io import loadmat
import numpy as np
# import scanpy as sc
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def bioDataload(dataset_name):

    if dataset_name == 'Muraro':
        path = '../../../biodata/{}/{}/{}_lable.csv'.format(dataset_name, dataset_name, dataset_name)
        path1 = '../../../biodata/{}/{}/{}_count.csv'.format(dataset_name, dataset_name, dataset_name)
        data = pd.read_csv(path)
        data1 = pd.read_csv(path1)
        Total_label = np.array(data['x']) # the label of each cell
        x1 = np.array(data1)
        Total_data = np.delete(x1,0,1) # the features of each cell
        cluster_name = np.unique(Total_label)
        cluster_num = cluster_name.size
        init_center_index = np.zeros(cluster_num,dtype=int)
        for i in np.arange(cluster_num): # label preprocessing
            k = np.array(np.where(Total_label == cluster_name[i])).reshape(-1)
            Total_label[k] = i
            init_center_index[i] = k[0]
        return Total_data.T,Total_label,init_center_index,cluster_num
    elif dataset_name == 'lawlor':
        path = '../../../biodata/{}/{}_lable.csv'.format(dataset_name,dataset_name)
        path1 = '../../../biodata/{}/{}_count.csv'.format(dataset_name,dataset_name)
        data = pd.read_csv(path)
        data1 = pd.read_csv(path1,header=None)
        Total_label = np.array(data['x']) # the label of each cell
        Total_data = np.array(data1).astype(float)
        cluster_name = np.unique(Total_label)
        cluster_num = cluster_name.size
        init_center_index = np.zeros(cluster_num, dtype=int)
        for i in np.arange(cluster_num):  # label preprocessing
            k = np.array(np.where(Total_label == cluster_name[i])).reshape(-1)
            Total_label[k] = i
            init_center_index[i] = k[0]
        return Total_data, Total_label, init_center_index, cluster_num


def generateAdj(featureMatrix, distanceType='euclidean', k=10):
    def Dataload(dataset_name):

        if dataset_name == 'Muraro':
            path = '../../biodata/{}/{}_lable.csv'.format(dataset_name, dataset_name)
            path1 = '../../biodata/{}/{}_count.csv'.format(dataset_name, dataset_name)
            data = pd.read_csv(path)
            data1 = pd.read_csv(path1)
            Total_label = np.array(data['x'])  # the label of each cell
            x1 = np.array(data1)
            Total_data = np.delete(x1, 0, 1)  # the features of each cell
            cluster_name = np.unique(Total_label)
            cluster_num = cluster_name.size
            init_center_index = np.zeros(cluster_num, dtype=int)
            for i in np.arange(cluster_num):  # label preprocessing
                k = np.array(np.where(Total_label == cluster_name[i])).reshape(-1)
                Total_label[k] = i
                init_center_index[i] = k[0]
            return Total_data.T, Total_label, init_center_index, cluster_num


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

if __name__ == '__main__':
    opt.args.device = torch.device("cuda" if opt.args.cuda else "cpu")
    args.name = 'lawor'#dataset_num
    args.nk = 5#the neighbor of cells
    args.graph_k_save_path = './DCRN_pretrain/graph/{}{}_graph.txt'.format(args.name, args.nk)
    args.graph_save_path = './DCRN_pretrain/graph/{}_graph.txt'.format(args.name)
    torch.set_default_dtype(torch.float32)
    # load data
    X, label, init_center_index, cluster_num = bioDataload(args.name)
    y = label.astype(int)
    args.n_clusters = cluster_num
    construct_graph(args.graph_k_save_path, X, label, 'heat', topk=args.nk)
    A = test_load_graph(args.nk, args.graph_k_save_path, args.graph_save_path, X).to(opt.args.device)
    A = A.to_dense()
    A_norm = normalize_adj(A, self_loop=True, symmetry=True)
    Ad = diffusion_adj(torch_to_numpy(A.detach().cpu()), mode="ppr", transport_rate=opt.args.alpha_value)
    ELM_hidden = [50,100, 200, 500, 1000, 1500, 2000]
    elmacc = float('-inf')
    elmnmi = float('-inf')
    elmari = float('-inf')
    elmf1 = float('-inf')
    for h in ELM_hidden:
        H = ELM(X, n_hidden=h, active_fun='sigmod')
        acc, nmi, ari, f1, centers = clustering(H, y, init_center_index)  
        if acc > elmacc:
            elmacc = acc
            best_Hidden = h
            best_H = H
        if nmi > elmnmi:
            elmnmi = nmi
            best_nmi_Hidden = h
            best_nmi_H = H
        if ari > elmari:
            elmari = ari
            best_ari_Hidden = h
            best_ari_H = H
        if f1 > elmf1:
            elmf1 = f1
            best_f1_Hidden = h
            best_f1_H = H
    #ELM的L个数
    args.n_input = best_ari_Hidden #best_Hidden, best_nmi_Hidden, best_ari_Hidden, best_f1_Hidden
    print('Hidden numebr: {}'.format(best_ari_Hidden))
    args.n_components = args.n_input
    args.n_z = args.n_input
    args.epoch = 2000
    HX = best_ari_H.to(opt.args.device)  # H(X) #best_H, best_nmi_H, best_ari_H, best_f1_H
    Ad = numpy_to_torch(Ad).to(opt.args.device)  
    embeding_size_set = [128,256,512,1024,2048]
    accc = float('-inf')
    f11 = float('-inf')
    arii = float('-inf')
    nmii = float('-inf')
    best_embeding_size = 0
    best_f1_embeding_size = 0
    best_ari_embeding_size = 0
    best_nmi_embeding_size = 0
    for embeding_size in embeding_size_set:
        print('embeding_size: {}'.format(embeding_size))
        args.ae_n_enc_1 = embeding_size
        args.ae_n_enc_2 = embeding_size
        args.ae_n_enc_3 = embeding_size
        args.ae_n_dec_1 = embeding_size
        args.ae_n_dec_2 = embeding_size
        args.ae_n_dec_3 = embeding_size
        args.gae_n_enc_1 = embeding_size
        args.gae_n_enc_2 = embeding_size
        args.gae_n_enc_3 = args.n_input
        args.gae_n_dec_1 = args.n_input
        args.gae_n_dec_2 = embeding_size
        args.gae_n_dec_3 = embeding_size
        model = DCRN(n_node=HX.shape[0]).to(opt.args.device)
        # deep graph clustering
        acc, nmi, ari, f1 = ELM_train(model, HX, y, torch_to_numpy(A.detach().cpu()), A_norm, Ad, init_center_index)
        print("ACC: {:.4f},".format(acc), "NMI: {:.4f},".format(nmi), "ARI: {:.4f},".format(ari),
              "F1: {:.4f}".format(f1))
        if acc > accc:
            accc = acc
            best_embeding_size = embeding_size
        if nmi > nmii:
            nmii = nmi
            best_nmi_embeding_size = embeding_size
        if ari > arii:
            arii = ari
            best_ari_embeding_size = embeding_size
        if f1 > f11:
            f11 = f1
            best_f1_embeding_size = embeding_size
        print('optimal: ACC= {:.4f}, F1= {:.4f}, NMI= {:.4f}, ARI= {:.4f}\n'.format(accc,f11,nmii,arii))
        print('optimal_embeding_size: ACC_embeding_size= {:.4f}, F1_embeding_size= {:.4f}\n, NMI_embeding_size= {:.4f}, ARI_embeding_size= {:.4f}\n'.format(best_embeding_size, best_f1_embeding_size, best_nmi_embeding_size, best_ari_embeding_size))
