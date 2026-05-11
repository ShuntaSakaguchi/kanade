import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution,Poisson, Gamma
from . import kanade_config

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nb_log_prob(x, mu, theta, eps): # scvi-toolのlog_nb_positive functinを参考にした
    log = torch.log
    lgamma = torch.lgamma
    log_theta_mu_eps = log(theta + mu + eps)

    res = (
        theta * (log(theta + eps) - log_theta_mu_eps)
        + x * (log(mu + eps) - log_theta_mu_eps)
        + lgamma(x + theta)
        - lgamma(theta)
        - lgamma(x + 1)
    )

    return res

def zinb_log_prob(x, mu, theta, pi, eps):
    log = torch.log
    lgamma = torch.lgamma

    gamma_term = lgamma(x+theta+eps)-lgamma(theta+eps)-lgamma(x + 1) #NBのガンマ分布が絡む部分のlog

    _zero_term = log(eps + pi + (1 - pi + eps) * torch.exp(gamma_term + theta * log((theta + eps)/ (mu + theta + eps))))
    zero_term = torch.where(x < eps, _zero_term, torch.tensor(0.0).to(kanade_config.device)) #zero

    _non_zero_term = log(1 - pi+eps) + gamma_term + x * (log(mu + eps) - log(mu + theta + eps)) + theta * (log(theta + eps) - log(mu + theta + eps))
    non_zero_term = torch.where(x > 0, _non_zero_term, torch.tensor(0.0).to(kanade_config.device)) #non-zero

    log_likelihood = zero_term + non_zero_term

    return log_likelihood


def gram_matrix_gaussian(X, sigma=1.0 , centered = True):

    K = torch.cdist(X, X)
    K = torch.exp(- K.pow(2)/(2*sigma**2))

    n = K.shape[0]

    if centered:
        ONE = torch.ones_like(K).to(kanade_config.device)
        Q = torch.eye(n=n, m=n).to(kanade_config.device) - ONE/n
        K = Q.mm(K).mm(Q)

    return K


def hsic(X, Y, sigma=1.0):

    Kx = gram_matrix_gaussian(X, sigma)
    Ky = gram_matrix_gaussian(Y, sigma)
    n = Kx.shape[0]
    val = torch.trace(Kx.mm(Ky)) / n**2

    return val

def convert_category(adata,col_name):
    _y1 = np.array(adata.obs[col_name])
    d = {x: i for i, x in enumerate(set(_y1))}
    y1 = [d[x] for x in _y1]

    return d, torch.tensor(y1, dtype=torch.long)
