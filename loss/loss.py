# -*- coding: utf-8 -*-
"""
@author: ByteHong
@organization: SCUT
"""
import torch

def pairwise_distances(x, y, power=2, sum_dim=2):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n,m,d)
    y = y.unsqueeze(0).expand(n,m,d)
    dist = torch.pow(x-y, power).sum(sum_dim)
    return dist

def StandardScaler(x,with_std=False):
    mean = x.mean(0, keepdim=True)
    std = x.std(0, unbiased=False, keepdim=True)
    x -= mean
    if with_std:
        x /= (std + 1e-10)
    return x



def RCS_loss(Xs,ys,Xt,yt0, DEVICE, lamda=1e-2, sigma=None,alpha=0.5):
    if alpha > 1.0 or alpha < 0.0:
        raise ValueError("alpha domain: 0.0<=alpha<=1.0")
    ns,dim = Xs.shape
    nt = Xt.shape[0]
    X = StandardScaler(torch.cat((Xs,Xt), dim=0))
    if sigma is None:
        pairwise_dist = pairwise_distances(Xs,Xt)
        sigma = torch.median(pairwise_dist[pairwise_dist!=0]).to(DEVICE)
    source_label = ys
    target_label = yt0
    n = ns + nt
    y = torch.cat((source_label, target_label), dim = 0)
    Ky = torch.tensor(y[:,None]==y, dtype=torch.float64).to(DEVICE)
    X_norm = torch.sum(X ** 2, axis=-1).to(DEVICE)
    K_W = torch.exp(-( X_norm[:, None] + X_norm[None,:] - 2 * torch.mm(X, X.T)) / sigma) * Ky
    H = float(alpha)/ns * torch.mm(K_W[:ns].T,K_W[:ns]) + float(1.0-alpha)/nt * torch.mm(K_W[ns:].T,K_W[ns:])
    # try to alter the regularization with K_w
    b = (torch.mean(K_W[:ns],axis=0)[:,None])
    theta = torch.mm((H + lamda * torch.eye(ns+nt).to(DEVICE)).inverse(), b)
    RCS = b.T @ theta - 0.5 * theta.T @ H @ theta #- 0.5
    return torch.mean(RCS)


