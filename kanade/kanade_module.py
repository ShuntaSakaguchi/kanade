import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution,Poisson, Gamma
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from anndata import AnnData

from .functions import *
from . import kanade_config

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dropout_rate = 0.1

class NegativeBinomial(Distribution):

    def __init__(self, mu, theta):

        super().__init__()
        self._eps = 1e-8
        self.theta = theta
        self.mu = mu
        self.gamma_dist = Gamma(theta, theta/(mu+self._eps))



    def sample(self, sample_shape=torch.Size()):

        lambda_ = self.gamma_dist.sample(sample_shape)
        return Poisson(lambda_).sample()

    def log_prob(self, value):

        return nb_log_prob(value, mu=self.mu, theta=self.theta, eps = self._eps)

    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        return self.mean + (self.mean**2) / self.theta


class ZINB(NegativeBinomial):

        def __init__(self, mu, theta, pi):
            super().__init__(mu = mu, theta = theta)
            self.pi = pi

        def sample(self, sample_shape=torch.Size()):
            sample_nb = super().sample(sample_shape=sample_shape)
            is_zero = torch.rand_like(sample_nb) <= self.pi
            sample = torch.where(is_zero, torch.zeros_like(sample_nb), sample_nb)

            return sample


        def log_prob(self, value):
            return zinb_log_prob(value, mu = self.mu, theta = self.theta, pi = self.pi, eps = self._eps)


        @property
        def mean(self):
            return (1-self.pi) * self.mu

        @property
        def variance(self):
            raise NotImplementedError



class Encoder(nn.Module):

    def __init__(self, input_dim, latent_dim, filter_sizes):
        super(Encoder, self).__init__()
        layers = []
        prev_dim = input_dim

        for size in filter_sizes:
            layers.append(nn.Linear(prev_dim, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = size

        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, latent_dim)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


class Decoder(nn.Module):

    def __init__(self, latent_dim, output_dim, filter_sizes):
        super(Decoder, self).__init__()
        layers = []
        prev_dim = latent_dim

        for size in list(reversed(filter_sizes)):
            layers.append(nn.Linear(prev_dim, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = size

        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)


class Decoder_ZINB(Decoder):

    def __init__(self, latent_dim, output_dim, filter_sizes, scale_max=1e6, scale_min=0):
        self.latent_dim = latent_dim
        super(Decoder_ZINB, self).__init__(latent_dim-1, output_dim, filter_sizes)
        # self.scale_decoder = nn.Linear(latent_dim, 1)
        self.theta_decoder = nn.Linear(latent_dim-1, 1)
        self._eps = 1e-8
        self.scale_logmax = np.log(scale_max)
        self.scale_logmin = np.log(scale_min)

    def forward(self, z):

        z_scale, z_main = z.split([1, self.latent_dim-1],dim=1)

        nn_output = self.model(z_main)
        batchsize = nn_output.shape[0]

        #出力をスケールファクター、NBの平均、NBの逆分散パラメータ、zero_inflated_probに分割
        _mu, _pi = nn_output.chunk(2, dim=1)
        mu = F.softmax(_mu, dim =1)

        pi = torch.sigmoid(_pi) #確率なので、0-1に制限

        z_scale_clamp = torch.clamp(z_scale, min=self.scale_logmin, max=self.scale_logmax)
        scale = z_scale_clamp.exp()

        _theta = self.theta_decoder(z_main)
        theta_pos = F.softplus(_theta) + self._eps #正の値に制限
        theta = torch.broadcast_to(theta_pos.view(batchsize, 1), (batchsize, mu.shape[1]))

        return mu, theta, pi, scale


class Predictor(nn.Module):

    def __init__(self, latent_dim, output_dim, predictor_type = 'classifier', legacy=False):

        super(Predictor, self).__init__()
        self.fc = nn.Linear(latent_dim, output_dim)

        self.legacy = legacy

        if predictor_type.upper() == 'REGRESSOR' or predictor_type.upper() == 'R':
            self.classifier = False

        elif predictor_type.upper() == 'CLASSIFIER' or predictor_type.upper() == 'C':
            self.classifier = True

        else:
            raise ValueError('Choose "predictor_type" from "classifier" or "regressor"')

    def forward(self, z):
        z = self.fc(z)
        #Previously, if it's classifier z was converted by Softmax.
        #But because input of F.cross_entropy is logit, we skip the softmax function.
        #Legacy method, which use softmax function, can be used by choosing 'legacy == True'.
        if self.legacy and self.classifier: z = F.softmax(z, dim=1)
        return z


class VAE_kanade(nn.Module):

    def __init__(self, input_dim, supervised_dim, unsupervised_dim, batch_dim, filter_sizes, dim_supervised_target, num_batch, predictor_type = 'classifier', scale_max=1e6, legacy=True):

        super(VAE_kanade, self).__init__()

        #check input types and shapes
        #check input_dim
        if not isinstance(input_dim, int):
            raise ValueError('"input_dim" must be int.')


        #check supervised_dim
        if isinstance(supervised_dim, int):
            supervised_dim = [supervised_dim]

        elif isinstance(supervised_dim, list):
            for d in supervised_dim:
                if not isinstance(d, int):
                    raise ValueError('"supervised_dim" must be int or list of ints.')

        else:
            raise ValueError('"supervised_dim" must be int or list of ints.')

        self.supervised_dim = supervised_dim


        #check unsupervised_dim
        if not isinstance(unsupervised_dim, int):
            raise ValueError('"unsupervised_dim" must be int.')

        self.unsupervised_dim = unsupervised_dim


        #check batch_dim
        if not isinstance(batch_dim, int):
            raise ValueError('"batch_dim" must be int.')

        self.batch_dim = batch_dim

        #check filter_sizes
        if not isinstance(filter_sizes, list):
            raise ValueError('"filter_sizes" must be list of ints.')

        else:
            for d in filter_sizes:
                if not isinstance(d, int):
                    raise ValueError('"supervised_dim" must be int or list of ints.')

        #check dim_supervised_target
        if isinstance(dim_supervised_target, int):

            dim_supervised_target = [dim_supervised_target] * len(dim_supervised_target)


        elif isinstance(dim_supervised_target, list):
            if len(self.supervised_dim)!=len(dim_supervised_target):
                raise ValueError('"supervised_dim" and "dim_supervised_target" must be the same length.')
            else:
                for d in dim_supervised_target:
                    if not isinstance(d, int):
                        raise ValueError('"dim_supervised_target" must be int or list of ints.')

        #check num_batch
        if not isinstance(num_batch, int):
            raise ValueError('"num_batch" must be int.')

        #check predictor_type
        if isinstance(predictor_type, str):
            if predictor_type.upper() in ['CLASSIFIER','REGRESSOR','C',"R"]:
                predictor_type = [predictor_type] * len(dim_supervised_target)
            else:
                raise ValueError('Choose predictor_type from "classifier" or "regressor".')

        elif isinstance(predictor_type, list):
            if len(self.supervised_dim)!=len(predictor_type):
                raise ValueError('"supervised_dim" and "predictor_type" must be the same length.')
            else:
                for t in predictor_type:
                    if not t.upper() in ['CLASSIFIER','REGRESSOR','C',"R"]:
                        raise ValueError('Choose predictor_type from "classifier" or "regressor".')


        #calculate the number of total dims
        total_dim = 1 + sum(supervised_dim) + unsupervised_dim + batch_dim

        self.encoder = Encoder(input_dim, total_dim, filter_sizes)
        self.decoder = Decoder_ZINB(total_dim, input_dim * 2, filter_sizes, scale_max=scale_max)

        self.supervised_predictors = []
        self.legacy = legacy # See comment in Predictor class
        for i,(d,n,t) in enumerate(zip(supervised_dim, dim_supervised_target, predictor_type)):
            p = Predictor(d, n, t, legacy=legacy)
            self.supervised_predictors.append(p)
            setattr(self,'predictor_{}'.format(i),p)


        self.batch_predictor = Predictor(batch_dim, num_batch, legacy=legacy)

    def forward(self, x):

        mu, log_var = self.encoder(x)

        split_length = [1, sum(self.supervised_dim), self.unsupervised_dim, self.batch_dim]

        mu_scale, mu_supervised, mu_unsupervised, mu_batch = mu.split(split_length, dim=1)
        log_var_scale, log_var_supervised, log_var_unsupervised, log_var_batch = log_var.split(split_length, dim=1)

        z_scale = reparameterize(mu_scale, log_var_scale)
        z_supervised = reparameterize(mu_supervised, log_var_supervised)
        if self.unsupervised_dim > 0:
            z_unsupervised = reparameterize(mu_unsupervised, log_var_unsupervised)
        z_batch = reparameterize(mu_batch, log_var_batch)


        supervised_pred = []
        for z_supervised_i, predictor in zip(z_supervised.split(self.supervised_dim, dim=1), self.supervised_predictors):
            pred_i = predictor(z_supervised_i)
            supervised_pred.append(pred_i)

        batch_pred = self.batch_predictor(z_batch)

        if self.unsupervised_dim > 0:
            z_concat = torch.cat((z_scale, z_supervised, z_unsupervised, z_batch), dim=1)

        else:
            z_concat = torch.cat((z_scale, z_supervised,z_batch), dim=1)

        mu_nb, theta, pi, scale = self.decoder(z_concat)

        return mu_nb, theta, pi, scale, supervised_pred, batch_pred, mu, log_var


    # def to(self, device):

    #     self = super(VAE_kanade, self).to(device)
    #     self.supervised_predictors = [p.to(device) for p in self.supervised_predictors]

    #     return self




class VAELoss_kanade(nn.Module):

    def __init__(self, independence_pattern = None, recon_loss_weight=1.0, kl_divergence_weight=1.0, predictor_loss_weight=10.0, batch_loss_weight = 10.0,independence_loss_weight=0, independence_criteria = 'Pearson', sigma = 1.0, predictor_type = 'classifier'):

        super(VAELoss_kanade, self).__init__()

        self.recon_loss_weight = recon_loss_weight
        self.kl_divergence_weight = kl_divergence_weight
        # self.predictor_loss_weight = predictor_loss_weight

        if isinstance(predictor_loss_weight, (int, float)):

            self.predictor_loss_weight = [predictor_loss_weight]

        elif isinstance(predictor_loss_weight, list):

            for w in predictor_loss_weight:
                if not isinstance(w, (int, float)):
                    raise ValueError('predictor_loss_weight must be a number or a list of numbers')

                self.predictor_loss_weight = predictor_loss_weight

        else:
            raise ValueError('predictor_loss_weight must be a number or a list of numbers')

        self.batch_loss_weight = batch_loss_weight


        self.independence_loss_weight = independence_loss_weight

        self.independence_pattern = independence_pattern

        # if independence_criteria.upper() == 'HSIC':
        #     self.independence_criteria_func = hsic(X[:,0].reshape(-1,1), X[:,1].reshape(-1,1), sigma = sigma)
        # elif independence_criteria.upper() == 'PEARSON':
        #     self.independence_criteria_func = lambda X: torch.corrcoef(torch.t(X))[0,1].pow(2)
        # else:
        #     raise ValueError('select the "independence_criteria" from Pearson or HSIC')

        self.independence_criteria = independence_criteria

        if isinstance(predictor_type, str):
            if predictor_type.upper() in ['CLASSIFIER','C']:
                self.predictor_loss_func = [F.cross_entropy]

            elif predictor_type.upper() in ['REGRESSOR','R']:
                self.predictor_loss_func = [F.mse_loss]

            else:
                raise ValueError('Choose predictor_type from "classifier" or "regressor".')

        elif isinstance(predictor_type, list):

            self.predictor_loss_func = []
            for t in predictor_type:
                if t.upper() in ['CLASSIFIER','C']:
                    self.predictor_loss_func.append(F.cross_entropy)

                elif t.upper() in ['REGRESSOR',"R"]:
                  self.predictor_loss_func.append(F.mse_loss)

                else:
                    raise ValueError('Choose predictor_type from "classifier" or "regressor".')
        else:
            raise ValueError('Choose predictor_type must be "classifier","regresser" or list of them')

    def forward(self, input, mu_nb, theta, pi, mu, log_var, supervised_pred, batch_pred, true_labels, true_batch):

        #initialize total loss
        total_loss = 0

        #reconstrution loss
        dist = ZINB(mu = mu_nb, theta = theta, pi = pi)
        self.recon_loss = -dist.log_prob(input).sum()
        self.recon_loss = self.recon_loss_weight * self.recon_loss

        total_loss += self.recon_loss



        # KL divergence loss
        self.kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        self.kl_divergence = self.kl_divergence_weight * self.kl_divergence

        total_loss += self.kl_divergence



        # classifier loss
        self.supervised_predictor_loss = 0

        #check predictor loss function
        if len(self.predictor_loss_func) == 1:
            predictor_loss_func = self.predictor_loss_func * len(supervised_pred)
        else:
            predictor_loss_func = self.predictor_loss_func

        #check predictor loss weight
        if len(self.predictor_loss_weight) == 1:
            predictor_loss_weight = self.predictor_loss_weight * len(supervised_pred)
        else:
            predictor_loss_weight = self.predictor_loss_weight


        #calculate classifier loss

        for pred, l, f, w in zip(supervised_pred, true_labels, predictor_loss_func, predictor_loss_weight):
            self.supervised_predictor_loss += w * f(pred, l)

        self.supervised_predictor_loss = self.supervised_predictor_loss * input.shape[0]
        total_loss += self.supervised_predictor_loss



        #batch predicor loss
        self.batch_predictor_loss = F.cross_entropy(batch_pred, true_batch)
        self.batch_predictor_loss = self.batch_loss_weight * self.batch_predictor_loss * input.shape[0]

        total_loss += self.batch_predictor_loss



        # independence loss
        self.independence_loss = 0

        mu_independence = mu.split(self.independence_pattern, dim=1)

        for mu_comb in itertools.combinations(mu_independence,2):

            if self.independence_criteria.upper() == 'PEARSON':

                mu_tmp = torch.cat(mu_comb, dim = 1)
                corr = torch.corrcoef(torch.t(mu_tmp)).pow(2)
                corr = corr[:mu_comb[0].shape[1],mu_comb[0].shape[1]:]
                self.independence_loss += corr.sum()

            elif self.independence_criteria.upper() == 'HSIC':

                self.independence_loss += hsic(mu_comb[0], mu_comb[1])

            else:
                raise ValueError('Choose independence_criteria from Pearson or HSIC.')

        self.independence_loss = self.independence_loss_weight * self.independence_loss


        total_loss += self.independence_loss



        return total_loss


class Trainer_kanade():

    def __init__(self, model, loss_func, optimizer, model_file_name, n_epochs = 100, logging =False, logfile_name_prefix = None):

        self.model = model
        self.loss_fn = loss_func
        self.optimizer = optimizer

        self.model_file_name = model_file_name

        self.n_epochs = n_epochs
        self.logging = logging

        if logfile_name_prefix is None:
            self.logfile_name_prefix = 'output_kanade'
        else:
            self.logfile_name_prefix = logfile_name_prefix


    def training(self, train_loader, test_loader = None, seed = None):


        if not seed is None:
            np.random.seed(seed_internal)
            torch.manual_seed(seed_internal)
            torch.cuda.manual_seed_all(seed_internal)


        logfile_name_train = self.logfile_name_prefix + '_train.log'

        if not test_loader is None:
            logfile_name_test = self.logfile_name_prefix + '_test.log'

        with open(logfile_name_train, "w") as f:
            pass

        with open(logfile_name_test, "w") as f:
            pass

        for epoch in tqdm(range(1, self.n_epochs + 1)):

            self.model.train()
            train_loss = 0
            recon_loss = 0
            kl_loss = 0
            supervised_pred_loss = 0
            batch_pred_loss = 0
            independence_loss = 0

            # training loop
            for batch_idx, loaded in enumerate(train_loader):

                loss = self.calc_loss(loaded)

                train_loss += loss.item()
                recon_loss += self.loss_fn.recon_loss
                kl_loss += self.loss_fn.kl_divergence
                supervised_pred_loss += self.loss_fn.supervised_predictor_loss
                batch_pred_loss += self.loss_fn.batch_predictor_loss
                independence_loss += self.loss_fn.independence_loss

                loss.backward()
                self.optimizer.step()


            with open(logfile_name_train, "a") as f:
                f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                    epoch,
                    train_loss / len(train_loader.dataset),
                    recon_loss / len(train_loader.dataset),
                    kl_loss / len(train_loader.dataset),
                    supervised_pred_loss / len(train_loader.dataset),
                    batch_pred_loss / len(train_loader.dataset),
                    independence_loss / len(train_loader.dataset),
                    )
                )


            if not test_loader is None:

                self.model.eval()
                test_loss = 0
                recon_loss = 0
                kl_loss = 0
                supervised_pred_loss = 0
                batch_pred_loss = 0
                independence_loss = 0

                #test loop
                with torch.no_grad():
                    for batch_idx, loaded in enumerate(test_loader):

                        loss = self.calc_loss(loaded)

                        test_loss += loss.item()
                        recon_loss += self.loss_fn.recon_loss
                        kl_loss += self.loss_fn.kl_divergence
                        supervised_pred_loss += self.loss_fn.supervised_predictor_loss
                        batch_pred_loss += self.loss_fn.batch_predictor_loss
                        independence_loss += self.loss_fn.independence_loss

               	with open(logfile_name_test, "a") as f:
                    f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                    epoch,
                    test_loss / len(test_loader.dataset),
                    recon_loss / len(test_loader.dataset),
                    kl_loss / len(test_loader.dataset),
                    supervised_pred_loss / len(test_loader.dataset),
                    batch_pred_loss / len(test_loader.dataset),
                    independence_loss / len(test_loader.dataset),
                    )
                )

            torch.save(self.model.state_dict(), self.model_file_name)

    def calc_loss(self,loaded):

        data = loaded[0].to(kanade_config.device)
        target_list = [l.to(kanade_config.device) for l in loaded[1:-1]]
        target_batch = loaded[-1].to(kanade_config.device)


        self.optimizer.zero_grad()

        mu_prob, theta, pi, scale, supervised_pred, batch_pred, mu, log_var = self.model(data)
        mu_nb = mu_prob * scale

        loss = self.loss_fn(
            data,
            mu_nb,
            theta,
            pi,
            mu,
            log_var,
            supervised_pred,
            batch_pred,
            target_list,
            target_batch
        )

        return loss



class DataLoader_kanade(DataLoader):

    def __init__(self, adata, supervised_columns, batch_column, predictor_type='classifier', batch_size=128, shuffle=True, seed=None, *args, **kwargs):

        if not isinstance(adata, AnnData):
            raise ValueError('"adata" must be AnnData object')


        if isinstance(supervised_columns, str):
            supervised_columns = [supervised_columns]

        if isinstance(predictor_type, str):

            if predictor_type.upper() in ['CLASSIFIER','C','REGRESSOR','R']:
               predictor_type = [predictor_type] * len(supervised_columns)

            else:
               raise ValueError('Choose predictor_type from "classifier" or "regressor".')

        X = np.array(adata.X.todense())
        X_tensor = torch.tensor(X, dtype=torch.float32)

        y_tensor_list = []
        self.y_map_dict = {}

        for l,t in zip(supervised_columns, predictor_type):
            if t.upper() in ['REGRESSOR','R']:
                y_ = adata.obs[l].astype(np.float32)
                y_tensor_ = torch.tensor(y_, dtype=torch.float32).reshape(-1,1)
                y_tensor_list.append(y_tensor_)

            else:
                y_map_, y_tensor_ = convert_category(adata,l)
                self.y_map_dict[l] = y_map_
                y_tensor_list.append(y_tensor_)

        y_map_, y_tensor_ = convert_category(adata,batch_column)
        self.y_map_dict['batch'] = y_map_
        y_tensor_list.append(y_tensor_)

        dataset_list = [X_tensor] + y_tensor_list
        dataset = TensorDataset(*dataset_list)

        super(DataLoader_kanade, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
