import torch
import tqdm
from utils import *
from torch.optim import Adam


def ELM_train(model, X, y, A, A_norm, Ad, init_center_index):
    """
    train our model
    Args:
        model: Dual Correlation Reduction Network
        X: input feature matrixscikit-learn
        y: input label
        A: input origin adj
        A_norm: normalized adj
        Ad: graph diffusion
    Returns: acc, nmi, ari, f1
    """
    print("Training…")
    # calculate embedding similarity and cluster centers
    sim, centers = ELM_model_init(X, y, init_center_index)

    # initialize cluster centers
    model.cluster_centers.data = torch.tensor(centers).to(opt.args.device)#类簇中心矩阵

    # edge-masked adjacency matrix (Am): remove edges based on feature-similarity
    Am = remove_edge(A, sim, remove_rate=0.1)
    best_acc_Z = torch.Tensor().to(opt.args.device)
    best_acc_y = torch.Tensor().to(opt.args.device)
    best_nmi_Z = torch.Tensor().to(opt.args.device)
    best_nmi_y = torch.Tensor().to(opt.args.device)
    best_ari_Z = torch.Tensor().to(opt.args.device)
    best_ari_y = torch.Tensor().to(opt.args.device)
    best_f1_Z = torch.Tensor().to(opt.args.device)
    best_f1_y = torch.Tensor().to(opt.args.device)
    optimizer = Adam(model.parameters(), lr=opt.args.lr)#模型求解器
    for epoch in tqdm.tqdm(range(opt.args.epoch)):#epoch一个训练周期
        # add gaussian noise to X
        X_tilde1, X_tilde2 = gaussian_noised_feature(X)

        # input & output
        X_hat, Z_hat, A_hat, _, Z_ae_all, Z_gae_all, Q, Z, AZ_all, Z_all = model(X_tilde1, Ad, X_tilde2, Am)#模型输出结果

        # calculate loss: L_{DICR}, L_{REC} and L_{KL}
        L_DICR = dicr_loss(Z_ae_all, Z_gae_all, AZ_all, Z_all)
        L_REC = reconstruction_loss(X, A_norm, X_hat, Z_hat, A_hat)
        L_KL = distribution_loss(Q, target_distribution(Q[0].data))
        loss = L_DICR + L_REC + opt.args.lambda_value * L_KL
        # optimization
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # loss.backward()
        optimizer.step()

        # clustering & evaluation
        with torch.no_grad():
            acc, nmi, ari, f1, _ = clustering(Z, y, init_center_index)#k-means聚类，Z最终的cell特征，y为每个cell的类簇标签
            if acc > opt.args.acc:
                opt.args.acc = acc
                best_acc_Z = Z
                best_acc_y = y
            if nmi > opt.args.nmi:
                opt.args.nmi = nmi
                best_nmi_Z = Z
                best_nmi_y = y
            if ari > opt.args.ari:
                opt.args.ari = ari
                best_ari_Z = Z
                best_ari_y = y
            if f1 > opt.args.f1:
                opt.args.f1 = f1
                best_f1_Z = Z
                best_f1_y = y

    return opt.args.acc, opt.args.nmi, opt.args.ari, opt.args.f1
    # return opt.args.acc, opt.args.nmi, opt.args.ari, opt.args.f1, \
    #     best_acc_Z, best_acc_y, best_nmi_Z, best_nmi_y, best_ari_Z, best_ari_y, best_f1_Z, best_f1_y
