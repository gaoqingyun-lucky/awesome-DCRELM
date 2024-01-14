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
    elif dataset_name == 'Bmcite':
        path = '../biodata/{}/{}_lable.csv'.format(dataset_name,dataset_name)
        path1 = '../biodata/{}/{}_count.csv'.format(dataset_name,dataset_name)
        data = pd.read_csv(path)
        data1 = pd.read_csv(path1)
        Total_label = np.array(data['x']) # the label of each cell
        x1 = np.array(data1)
        Total_data = np.delete(x1, 0, 1)
        cluster_name = np.unique(Total_label)
        cluster_num = cluster_name.size
        init_center_index = np.zeros(cluster_num, dtype=int)
        for i in np.arange(cluster_num):  # label preprocessing
            k = np.array(np.where(Total_label == cluster_name[i])).reshape(-1)
            Total_label[k] = i
            init_center_index[i] = k[0]
        return Total_data.T.astype(float), Total_label, init_center_index, cluster_num
    elif dataset_name == 'Yeo':
        file = '../../../biodata/{}.mat'.format(dataset_name)
        # mat_dtype=True，保证了导入后变量的数据类型与原类型一致。
        data = loadmat(file, mat_dtype=True)
        t = data['true_labs'].reshape(-1)
        Total_label = data['true_labs'].reshape(-1).astype(int) # the label of each cell
        Total_data = data['in_X'].astype(float)
        cluster_name = np.unique(data['true_labs'])
        cluster_num = cluster_name.size
        init_center_index = np.zeros(cluster_num, dtype=int)
        for i in np.arange(cluster_num):  # label preprocessing
            k = np.array(np.where(Total_label == cluster_name[i])).reshape(-1)
            Total_label[k] = i
            init_center_index[i] = k[0]
        return Total_data, Total_label, init_center_index, cluster_num
    elif dataset_name == 'pomeroy':
        file = '../../../biodata/{}.mat'.format(dataset_name)
        # mat_dtype=True，保证了导入后变量的数据类型与原类型一致。
        data = loadmat(file, mat_dtype=True)
        t = data['true_labs'].reshape(-1)
        Total_label = data['true_labs'].reshape(-1).astype(int) # the label of each cell
        Total_data = data['in_X'].astype(float)
        cluster_name = np.unique(data['true_labs'])
        cluster_num = cluster_name.size
        init_center_index = np.zeros(cluster_num, dtype=int)
        for i in np.arange(cluster_num):  # label preprocessing
            k = np.array(np.where(Total_label == cluster_name[i])).reshape(-1)
            Total_label[k] = i
            init_center_index[i] = k[0]
        return Total_data, Total_label, init_center_index, cluster_num
    elif dataset_name == 'Chung':
        file = '../../../biodata/{}.mat'.format(dataset_name)
        # mat_dtype=True，保证了导入后变量的数据类型与原类型一致。
        data = loadmat(file, mat_dtype=True)
        t = data['true_labs'].reshape(-1)
        Total_label = data['true_labs'].reshape(-1).astype(int) # the label of each cell
        Total_data = data['in_X'].astype(float)
        cluster_name = np.unique(data['true_labs'])
        cluster_num = cluster_name.size
        init_center_index = np.zeros(cluster_num, dtype=int)
        for i in np.arange(cluster_num):  # label preprocessing
            k = np.array(np.where(Total_label == cluster_name[i])).reshape(-1)
            Total_label[k] = i
            init_center_index[i] = k[0]
        return Total_data, Total_label, init_center_index, cluster_num
    elif dataset_name == 'Ning':
        file = '../../../biodata/{}.mat'.format(dataset_name)
        # mat_dtype=True，保证了导入后变量的数据类型与原类型一致。
        data = loadmat(file, mat_dtype=True)
        t = data['true_labs'].reshape(-1)
        Total_label = data['true_labs'].reshape(-1).astype(int) # the label of each cell
        Total_data = data['in_X'].astype(float)
        cluster_name = np.unique(data['true_labs'])
        cluster_num = cluster_name.size
        init_center_index = np.zeros(cluster_num, dtype=int)
        for i in np.arange(cluster_num):  # label preprocessing
            k = np.array(np.where(Total_label == cluster_name[i])).reshape(-1)
            Total_label[k] = i
            init_center_index[i] = k[0]
        return Total_data, Total_label, init_center_index, cluster_num
    elif dataset_name == 'Test_human':
        file = '../../../biodata/{}.mat'.format(dataset_name)
        # mat_dtype=True，保证了导入后变量的数据类型与原类型一致。
        data = loadmat(file, mat_dtype=True)
        t = data['true_labs'].reshape(-1)
        Total_label = data['true_labs'].reshape(-1).astype(int) # the label of each cell
        Total_data = data['in_X'].astype(float)
        cluster_name = np.unique(data['true_labs'])
        cluster_num = cluster_name.size
        init_center_index = np.zeros(cluster_num, dtype=int)
        for i in np.arange(cluster_num):  # label preprocessing
            k = np.array(np.where(Total_label == cluster_name[i])).reshape(-1)
            Total_label[k] = i
            init_center_index[i] = k[0]
        return Total_data, Total_label, init_center_index, cluster_num
    elif dataset_name == 'Test_1_Zeisel_big':
        file = '../../../biodata/{}.mat'.format(dataset_name)
        # mat_dtype=True，保证了导入后变量的数据类型与原类型一致。
        data = loadmat(file, mat_dtype=True)
        # t = data['true_labs'].reshape(-1)
        Total_label = data['true_labs'].reshape(-1).astype(int) # the label of each cell
        Total_data = data['in_X'].A.astype(float)
        cluster_name = np.unique(data['true_labs'])
        cluster_num = cluster_name.size
        init_center_index = np.zeros(cluster_num, dtype=int)
        for i in np.arange(cluster_num):  # label preprocessing
            k = np.array(np.where(Total_label == cluster_name[i])).reshape(-1)
            Total_label[k] = i
            init_center_index[i] = k[0]
        return Total_data, Total_label, init_center_index, cluster_num
    elif dataset_name == 'Camp1':
        path = '../../../biodata/{}'.format(dataset_name)
        # path = '../../../../../biodata/Camp1'
        data = []
        file = open(path, 'r')  # 打开文件
        file_data = file.readlines()  # 读取所有行
        for row in file_data:
            tmp_list = row.split('\t')  # 按‘，’切分每行的数据
            # tmp_list[-1] = tmp_list[-1].replace('\n',',') #去掉换行符
            data.append(tmp_list)  # 将每行数据插入data中
        data = np.array(data)
        data1 = np.delete(data, 0, 0)
        Total_data = data1[:, 1:-1]
        Total_label = data1[:, -1]
        cluster_name = np.unique(Total_label)
        cluster_num = cluster_name.size
        init_center_index = np.zeros(cluster_num, dtype=int)
        for i in np.arange(cluster_num):  # label preprocessing
            k = np.array(np.where(Total_label == cluster_name[i])).reshape(-1)
            Total_label[k] = i
            init_center_index[i] = k[0]
        return Total_data.astype(float), Total_label.astype(int), init_center_index, cluster_num
    elif dataset_name == 'kolo':
        path = '../../../biodata/{}/{}_lable.csv'.format(dataset_name, dataset_name)
        path1 = '../../../biodata/{}/{}_pre.csv'.format(dataset_name, dataset_name)
        data = pd.read_csv(path)
        data1 = pd.read_csv(path1, header=None)
        Total_label = np.array(data['x'])  # the label of each cell
        Total_data = np.array(data1).astype(float)
        cluster_name = np.unique(Total_label)
        cluster_num = cluster_name.size
        init_center_index = np.zeros(cluster_num, dtype=int)
        for i in np.arange(cluster_num):  # label preprocessing
            k = np.array(np.where(Total_label == cluster_name[i])).reshape(-1)
            Total_label[k] = i
            init_center_index[i] = k[0]
        return Total_data.astype(float), Total_label.astype(int), init_center_index, cluster_num
    elif dataset_name == 'Klein':
        path = '../../../biodata/{}/{}_cell_label.csv'.format(dataset_name, dataset_name)
        path1 = '../../../biodata/{}/T2000_expression.txt'.format(dataset_name)
        data = pd.read_csv(path)
        data1 = []
        file = open(path1, 'r')  # 打开文件
        file_data = file.readlines()  # 读取所有行
        for row in file_data:
            tmp_list = row.split('\t')  # 按‘，’切分每行的数据
            # tmp_list[-1] = tmp_list[-1].replace('\n',',') #去掉换行符
            data1.append(tmp_list)  # 将每行数据插入data中
        data1 = np.array(data1)
        data1 = np.delete(data1, 0, 0)
        data1 = np.delete(data1, 0, 1)
        # data1 = pd.read_csv(path1,header=None)
        Total_label = np.array(data['Cluster']).astype(int)  # the label of each cell
        Total_data = np.array(data1).astype(float).T
        cluster_name = np.unique(Total_label)
        cluster_num = cluster_name.size
        init_center_index = np.zeros(cluster_num, dtype=int)
        for i in np.arange(cluster_num):  # label preprocessing
            k = np.array(np.where(Total_label == cluster_name[i])).reshape(-1)
            Total_label[k] = i
            init_center_index[i] = k[0]
        return Total_data.astype(float), Total_label.astype(int), init_center_index, cluster_num
    elif dataset_name == 'Kolodziejczyk':
        path = '../../../biodata/{}/{}_cell_label.csv'.format(dataset_name, dataset_name)
        path1 = '../../../biodata/{}/T2000_expression.txt'.format(dataset_name)
        data = pd.read_csv(path)
        data1 = []
        file = open(path1, 'r')  # 打开文件
        file_data = file.readlines()  # 读取所有行
        for row in file_data:
            tmp_list = row.split('\t')  # 按‘，’切分每行的数据
            # tmp_list[-1] = tmp_list[-1].replace('\n',',') #去掉换行符
            data1.append(tmp_list)  # 将每行数据插入data中
        data1 = np.array(data1)
        data1 = np.delete(data1, 0, 0)
        data1 = np.delete(data1, 0, 1)
        # data1 = pd.read_csv(path1,header=None)
        Total_label = np.array(data['Cluster']).astype(int)  # the label of each cell
        Total_data = np.array(data1).astype(float).T
        cluster_name = np.unique(Total_label)
        cluster_num = cluster_name.size
        init_center_index = np.zeros(cluster_num, dtype=int)
        for i in np.arange(cluster_num):  # label preprocessing
            k = np.array(np.where(Total_label == cluster_name[i])).reshape(-1)
            Total_label[k] = i
            init_center_index[i] = k[0]
        return Total_data, Total_label, init_center_index, cluster_num
    elif dataset_name == 'BEAM':
        path = r'../../../biodata/BEAM-Ab_Mouse_HEL/cluster_label.csv'
        path1 = r'../../../biodata/BEAM-Ab_Mouse_HEL/matrix.mtx'
        dataset = scipy.io.mmread(path1)
        # adata = sc.read(path1)
        # dataset = adata.X
        Total_data = np.array(dataset.todense()).astype(float).T
        Cluster_label = pd.read_csv(path)
        Total_label = np.array(Cluster_label['Cluster']).astype(int)
        Total_label = Total_label -1
        cluster_name = np.unique(Total_label)
        cluster_num = cluster_name.size
        init_center_index = np.zeros(cluster_num, dtype=int)
        for i in np.arange(cluster_num):  # label preprocessing-
            k = np.array(np.where(Total_label == cluster_name[i])).reshape(-1)
            # Total_label[k] = i
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
        elif dataset_name == 'Biase':
            path = '../../biodata/{}/{}_lable.csv'.format(dataset_name, dataset_name)
            path1 = '../../biodata/{}/{}_count.csv'.format(dataset_name, dataset_name)
            data = pd.read_csv(path)
            data1 = pd.read_csv(path1, header=None)
            Total_label = np.array(data['x'])  # the label of each cell
            Total_data = np.array(data1).astype(float)
            cluster_name = np.unique(Total_label)
            cluster_num = cluster_name.size
            init_center_index = np.zeros(cluster_num, dtype=int)
            for i in np.arange(cluster_num):  # label preprocessing
                k = np.array(np.where(Total_label == cluster_name[i])).reshape(-1)
                Total_label[k] = i
                init_center_index[i] = k[0]
            return Total_data, Total_label, init_center_index, cluster_num
        elif dataset_name == 'Chung':
            path = '../../../../../biodata/{}'.format(dataset_name)
            # path = '../../../../../biodata/Camp1'
            data = []
            file = open(path, 'r')  # 打开文件
            file_data = file.readlines()  # 读取所有行
            for row in file_data:
                tmp_list = row.split('\t')  # 按‘，’切分每行的数据
                # tmp_list[-1] = tmp_list[-1].replace('\n',',') #去掉换行符
                data.append(tmp_list)  # 将每行数据插入data中
            data = np.array(data)
            data1 = np.delete(data, 0, 0)
            Total_data = data1[:, 1:-1]
            Total_label = data1[:, -1]
            cluster_name = np.unique(Total_label)
            cluster_num = cluster_name.size
            init_center_index = np.zeros(cluster_num, dtype=int)
            for i in np.arange(cluster_num):  # label preprocessing
                k = np.array(np.where(Total_label == cluster_name[i])).reshape(-1)
                Total_label[k] = i
                init_center_index[i] = k[0]
            return Total_data, Total_label, init_center_index, cluster_num
        elif dataset_name == 'kolo':
            path = '../../../../../biodata/{}/{}_lable.csv'.format(dataset_name, dataset_name)
            path1 = '../../../../../biodata/{}/{}_pre.csv'.format(dataset_name, dataset_name)
            data = pd.read_csv(path)
            data1 = pd.read_csv(path1, header=None)
            Total_label = np.array(data['x'])  # the label of each cell
            Total_data = np.array(data1).astype(float)
            cluster_name = np.unique(Total_label)
            cluster_num = cluster_name.size
            init_center_index = np.zeros(cluster_num, dtype=int)
            for i in np.arange(cluster_num):  # label preprocessing
                k = np.array(np.where(Total_label == cluster_name[i])).reshape(-1)
                Total_label[k] = i
                init_center_index[i] = k[0]
            return Total_data, Total_label, init_center_index, cluster_num
        elif dataset_name == 'Klein':
            path = '../../../../../biodata/{}/{}_cell_label.csv'.format(dataset_name, dataset_name)
            path1 = '../../../../../biodata/{}/T2000_expression.txt'.format(dataset_name)
            data = pd.read_csv(path)
            data1 = []
            file = open(path1, 'r')  # 打开文件
            file_data = file.readlines()  # 读取所有行
            for row in file_data:
                tmp_list = row.split('\t')  # 按‘，’切分每行的数据
                # tmp_list[-1] = tmp_list[-1].replace('\n',',') #去掉换行符
                data1.append(tmp_list)  # 将每行数据插入data中
            data1 = np.array(data1)
            data1 = np.delete(data1, 0, 0)
            data1 = np.delete(data1, 0, 1)
            # data1 = pd.read_csv(path1,header=None)
            Total_label = np.array(data['Cluster']).astype(int)  # the label of each cell
            Total_data = np.array(data1).astype(float).T
            cluster_name = np.unique(Total_label)
            cluster_num = cluster_name.size
            init_center_index = np.zeros(cluster_num, dtype=int)
            for i in np.arange(cluster_num):  # label preprocessing
                k = np.array(np.where(Total_label == cluster_name[i])).reshape(-1)
                Total_label[k] = i
                init_center_index[i] = k[0]
            return Total_data, Total_label, init_center_index, cluster_num
        elif dataset_name == 'Bmcite':
            path = '../../biodata/{}/{}_lable.csv'.format(dataset_name, dataset_name)
            path1 = '../../biodata/{}/{}_count.csv'.format(dataset_name, dataset_name)
            data = pd.read_csv(path)
            data1 = pd.read_csv(path1)
            Total_label = np.array(data['x'])  # the label of each cell
            x1 = np.array(data1)
            Total_data = np.delete(x1, 0, 1)
            cluster_name = np.unique(Total_label)
            cluster_num = cluster_name.size
            init_center_index = np.zeros(cluster_num, dtype=int)
            for i in np.arange(cluster_num):  # label preprocessing
                k = np.array(np.where(Total_label == cluster_name[i])).reshape(-1)
                Total_label[k] = i
                init_center_index[i] = k[0]
            return Total_data.T.astype(float), Total_label, init_center_index, cluster_num
    distMat = distance.cdist(featureMatrix,featureMatrix, distanceType)
    edgeList = np.zeros((featureMatrix.shape[0],featureMatrix.shape[0]), dtype=int)
    for i in np.arange(distMat.shape[0]):
        res = distMat[:,i].argsort()[:k]
        edgeList[i,res] = 1
    return edgeList

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
    # setup0
    # setup()
    opt.args.device = torch.device("cuda" if opt.args.cuda else "cpu")
    args.name = 'Klein'#数据集名字
    args.nk = 5#细胞的邻居
    args.graph_k_save_path = './DCRN_pretrain/graph/{}{}_graph.txt'.format(args.name, args.nk)#细胞graph存储的路径
    args.graph_save_path = './DCRN_pretrain/graph/{}_graph.txt'.format(args.name)
    # args.n_input = 10
    torch.set_default_dtype(torch.float32)#设置tensor的数据类型float32
    # Bio_data
    #导入数据
    X, label, init_center_index, cluster_num = bioDataload(args.name)#X是cell-RNA矩阵，label是类簇标签，init_center_index初始的类簇中心，cluster_num是类簇个数
    # X = (X - np.min(X)) / (np.max(X) - np.min(X))  # 归一化
    # args.n_input = X.shape[1]
    # args.n_components = X.shape[1]
    # args.n_z = X.shape[1]
    # args.gae_n_enc_3 = args.n_input
    # args.gae_n_dec_1 = args.n_input
    y = label.astype(int)#cell的类簇标签转化成整数型
    args.n_clusters = cluster_num#参数n_clusters设置为类簇个数
    construct_graph(args.graph_k_save_path, X, label, 'heat', topk=args.nk)
    A = test_load_graph(args.nk, args.graph_k_save_path, args.graph_save_path, X).to(opt.args.device)#cell相似性矩阵
    A = A.to_dense()#cell相似性矩阵转化成稠密形式
    A_norm = normalize_adj(A, self_loop=True, symmetry=True)
    Ad = diffusion_adj(torch_to_numpy(A.detach().cpu()), mode="ppr", transport_rate=opt.args.alpha_value)#cell相似性矩阵的diffusion
    # pca = PCA(n_components=args.n_input)
    # X = pca.fit_transform(X)
    # to torch tensor
    # X = numpy_to_torch(X).to(opt.args.device)
    ELM_hidden = [50,100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    # ELM_hidden = [1500]#换100, 200, 500, 1000, 1500, 2000
    elmacc = float('-inf')
    elmnmi = float('-inf')
    elmari = float('-inf')
    elmf1 = float('-inf')
    for h in ELM_hidden:
        H = ELM(X, n_hidden=h, active_fun='sigmod')
        acc, nmi, ari, f1, centers = clustering(H, y, init_center_index)  # k-means聚类，centers是类簇中心
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
    args.n_input = best_ari_Hidden #修改，best_Hidden, best_nmi_Hidden, best_ari_Hidden, best_f1_Hidden
    print('Hidden numebr: {}'.format(best_ari_Hidden))
    args.n_components = args.n_input
    args.n_z = args.n_input
    args.epoch = 2000
    # H = ELM(X, n_hidden=args.n_input, active_fun='sigmod')
    HX = best_ari_H.to(opt.args.device)  #修改 H(X) #best_H, best_nmi_H, best_ari_H, best_f1_H
    # A_norm = numpy_to_torch(A_norm, sparse=True).to(opt.args.device)
    Ad = numpy_to_torch(Ad).to(opt.args.device)  # Ad传输到GPU 'cuda'

    # embeding_size = 2048
    embeding_size_set = [2048]#128,256,512,1024,2048,4096
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

        args.ae_n_enc_1 = embeding_size#AE的第一层encoding维度
        args.ae_n_enc_2 = embeding_size#AE的第二层encoding维度
        args.ae_n_enc_3 = embeding_size#AE的第二层encoding维度
        args.ae_n_dec_1 = embeding_size#AE的第一层decoding维度
        args.ae_n_dec_2 = embeding_size
        args.ae_n_dec_3 = embeding_size
        args.gae_n_enc_1 = embeding_size#IGAE的第一层encoding维度
        args.gae_n_enc_2 = embeding_size
        args.gae_n_enc_3 = args.n_input
        args.gae_n_dec_1 = args.n_input#IGAE的第一层decoding维度
        args.gae_n_dec_2 = embeding_size
        args.gae_n_dec_3 = embeding_size


        # # Dual Correlation Reduction Network
        # model = DCRN(n_node=HX.shape[0]).to(opt.args.device)
        # # deep graph clustering
        # # acc, nmi, ari, f1 = train(model, X, y, torch_to_numpy(A.detach().cpu()), A_norm, Ad, init_center_index)
        # acc, nmi, ari, f1 = ELM_train(model, HX, y, torch_to_numpy(A.detach().cpu()), A_norm, Ad, init_center_index)
        # print("ACC: {:.4f},".format(acc), "NMI: {:.4f},".format(nmi), "ARI: {:.4f},".format(ari), "F1: {:.4f}".format(f1))
        # cycle = 1  # the number of round

        # for i in range(cycle):
        # print("Round {}".format(i))
        # Dual Correlation Reduction Network
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
