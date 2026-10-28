import numpy as np
import datetime
import pickle

from sklearn.model_selection import train_test_split

import torch
import torch.optim as optim
import torch.nn.functional as F

from .kanade_module import *
from . import kanade_config #ここでimportすると、外部から操作困難？


class Kanade():

    def __init__(self, adata, supervised_columns, batch_column, dim_supervised_target, num_batch, supervised_dim, predictor_type = None, unsupervised_dim=0, batch_dim=10, filter_sizes=None, do_train_test_split=True, test_ratio=0.1, split_seed=0, learning_seed=0, loss_fn_weights = None, mini_batch_size=128, device=None, kwargs_vae = None, kwargs_loss_fn=None, do_parallel=False, legacy=False, *args, **kwargs):

        self.input_dim = adata.shape[1]
        self.supervised_dim = supervised_dim
        self.unsupervised_dim = unsupervised_dim

        self.supervised_columns = supervised_columns
        self.batch_column = batch_column

        self.dim_supervised_target = dim_supervised_target
        self.num_batch = num_batch

        if isinstance(supervised_dim, int): supervised_dim = [supervised_dim] * len(supervised_columns)
        self.supervised_dim = supervised_dim

        self.batch_dim = batch_dim

        self.dim_dict = {'scale':slice(0,1,1)}
        idx = 1
        for colname, n_dimension in zip(supervised_columns, supervised_dim):

            self.dim_dict[colname] = slice(idx, idx+n_dimension,1)
            idx = idx+n_dimension

        if unsupervised_dim > 0:

            self.dim_dict['unsupervised_dim'] = slice(idx, idx+unsupervised_dim,1)

        idx = idx+unsupervised_dim

        self.dim_dict['batch_effect'] = slice(idx, idx+batch_dim, 1)

        self.legacy=legacy #See comment in Predictor class

        if predictor_type is None:
            self.predictor_type = ['c'] * len(self.supervised_columns)

        elif isinstance(predictor_type, str):
            if predictor_type.lower() in ['c', 'r', 'classifier', 'regressor']:
                self.predictor_type = [predictor_type] * len(self.supervised_columns)

            else:
                raise ValueError('Choose predictor_type from "classifier" or "regressor".')

        elif isinstance(predictor_type, list):

            if not len(predictor_type)==len(self.supervised_columns):
                raise ValueError('Length of "predictor_type" must be the same as that of "supervised_columns".')

            for ptype in predictor_type:
                if not ptype.lower() in ['c', 'r', 'classifier', 'regressor']:
                    raise ValueError('Choose predictor_type from "classifier" or "regressor".')
            self.predictor_type = predictor_type

        else:
            raise ValueError('Choose predictor_type from "classifier" or "regressor".')

        if filter_sizes is None:
            filter_sizes = [2000,400,200]

        self.filter_sizes = filter_sizes


        self.model_file = None
        self.log_file = None

        self.do_train_test_split = do_train_test_split
        self.test_ratio = test_ratio
        self.split_seed = split_seed



        self.train_loader = None
        self.test_loader = None

        self.learning_seed = learning_seed

        self.file_prefix = self._generate_file_prefix()


        self.mini_batch_size = mini_batch_size

        if device is None:
            device = kanade_config.device

        self.device = device

        self.kwargs_vae = kwargs_vae
        self.kwargs_loss_fn = kwargs_loss_fn
        self.do_parallel = do_parallel


        torch.manual_seed(learning_seed)
        torch.cuda.manual_seed_all(learning_seed)

        if not device is None:
            kanade_config.device = device

        self.generate_kanade_loader(adata, supervised_columns, unsupervised_dim, batch_column, self.predictor_type, filter_sizes, mini_batch_size, do_train_test_split, test_ratio, split_seed)

        self.model = self.generate_vae(adata.shape[1], supervised_dim, dim_supervised_target, unsupervised_dim, batch_dim, num_batch, self.predictor_type, filter_sizes, legacy, kwargs_vae, do_parallel)

        self.loss_fn = self.generate_loss_function(supervised_dim, unsupervised_dim, batch_dim, self.predictor_type, kwargs_loss_fn)


    def train_test_split(self, adata, test_ratio=0.1, split_seed=0):

        train_index, test_index = train_test_split(range(adata.shape[0]), test_size = test_ratio, random_state = split_seed)
        self.train_index = train_index
        self.test_index = test_index


    def generate_kanade_loader(self, adata, supervised_columns, unsupervised_dim, batch_column, predictor_type, filter_sizes, mini_batch_size, do_train_test_split=True, test_ratio=0.1, split_seed=0):

        if do_train_test_split:
            self.train_test_split(adata, test_ratio, split_seed)

            self.train_loader = DataLoader_kanade(
                adata = adata[self.train_index,:],
                supervised_columns = supervised_columns,
                batch_column = batch_column,
                predictor_type = predictor_type,
                batch_size = mini_batch_size,
                shuffle=True,
                seed=None
                )

            self.test_loader = DataLoader_kanade(
                adata = adata[self.test_index,:],
                supervised_columns = supervised_columns,
                batch_column = batch_column,
                predictor_type = predictor_type,
                batch_size = mini_batch_size,
                shuffle=True,
                y_map_dict = self.train_loader.y_map_dict,
                seed=None
                )

        else:
            self.train_loader = DataLoader_kanade(
                adata = adata,
                supervised_columns = supervised_columns,
                batch_column = batch_column,
                predictor_type = predictor_type,
                batch_size = mini_batch_size,
                shuffle=True,
                seed=None
                )


    def generate_vae(self, input_dim, supervised_dim, dim_supervised_target, unsupervised_dim, batch_dim, num_batch, predicor_type, filter_sizes,legacy, kwargs_vae, do_parallel=True):

        if kwargs_vae is None:
            kwargs_vae = {}

        model = VAE_kanade(
            input_dim = input_dim,
            supervised_dim = supervised_dim,
            unsupervised_dim = unsupervised_dim,
            batch_dim = batch_dim,
            dim_supervised_target = dim_supervised_target,
            num_batch = num_batch,
            predictor_type = predicor_type,
            filter_sizes = filter_sizes,
            legacy=legacy,
            **kwargs_vae
            ).to(kanade_config.device)

        if do_parallel:

            if kanade_config.device == torch.device("cuda"):
                model = torch.nn.DataParallel(model)  # make parallel
                torch.backends.cudnn.benchmark = True

            else:
                raise UserWarning('The selected device is not set to "cuda". The model is prepaired in non-parallel mode. \nTo enable parallel mode, please select "cuda" as your device.')

        return model


    def generate_loss_function(self, supervised_dim, unsupervised_dim, batch_dim, predicor_type, kwargs_loss_fn=None):

        if kwargs_loss_fn is None:
            kwargs_loss_fn = {}

        if isinstance(supervised_dim, int):
            supervised_dim = [supervised_dim]

        if unsupervised_dim > 0:

            independence_pattern = [1] + supervised_dim + [unsupervised_dim] + [batch_dim]   #scale factor + supervised_dims + unsupervised_dims + batch_dim

        else:

            independence_pattern = [1] + supervised_dim + [batch_dim] #scale factor + supervised_dims + batch_dim

        loss_fn = VAELoss_kanade(
            independence_pattern = independence_pattern,
            predictor_type = predicor_type,
            **kwargs_loss_fn
            ).to(kanade_config.device)

        return loss_fn

    def training(self, model_file=None,log_file_prefix=None, optimizer = None, logging=True, n_epochs=50, learning_rate=0.001):

        if model_file is None and log_file_prefix is None:

            file_prefix = self.file_prefix

        if model_file is None:
            self.model_file = file_prefix+'.pth'

        else:
            self.model_file = model_file

        if log_file_prefix is None:
            if not model_file is None:
                self.log_file = model_file

            else:
                self.log_file = file_prefix

        else:
            self.log_file = log_file_prefix


        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()

        trainer = Trainer_kanade(
            model = self.model,
            loss_func = self.loss_fn,
            optimizer = optimizer,
            model_file_name = self.model_file,
            n_epochs = n_epochs,
            logging = logging,
            logfile_name_prefix = self.log_file
            )

        trainer.training(train_loader = self.train_loader, test_loader = self.test_loader)

        self.model.eval()


    def projection(self, X, mini_batch_size=1024, temporary_device='cpu', return_mu_log_var=False):

        X = np.array(X.todense())

        X_tensor = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset,shuffle=False, batch_size=mini_batch_size)

        z_list = []
        mu_list = []
        log_var_list = []
        with torch.no_grad():
            for minibatch in dataloader:
                mu, log_var = self.model.encoder(minibatch[0].to(kanade_config.device))
                z = reparameterize(mu, log_var)
                z_list.append(z.to(temporary_device))
                mu_list.append(mu.to(temporary_device))
                log_var_list.append(log_var.to(temporary_device))

        mu = torch.cat(mu_list, dim=0)
        log_var = torch.cat(log_var_list, dim=0)
        z = torch.cat(z_list, dim=0)

        del X_tensor
        torch.cuda.empty_cache()

        if return_mu_log_var:
            return z, mu, log_var

        else:
            return z

    def decode(self, z, mini_batch_size=1024, temporary_device='cpu'):

        z_tensor = torch.tensor(z, dtype=torch.float32)
        dataset = TensorDataset(z_tensor)
        dataloader = DataLoader(dataset,shuffle=False, batch_size=mini_batch_size)

        mu_nb_list = []
        theta_list = []
        pi_list = []
        scale_list = []

        with torch.no_grad():
            for minibatch in dataloader:
                mu_nb, theta, pi, scale = self.model.decoder(minibatch[0].to(kanade_config.device))
                mu_nb_list.append(mu_nb.to(temporary_device))
                theta_list.append(theta.to(temporary_device))
                pi_list.append(pi.to(temporary_device))
                scale_list.append(scale.to(temporary_device))

        mu_nb = torch.cat(mu_nb_list, dim=0)
        theta = torch.cat(theta_list, dim=0)
        pi = torch.cat(pi_list, dim=0)
        scale = torch.cat(scale_list, dim=0)

        return mu_nb, theta, pi, scale

    def resample(self, mu_nb, theta, pi, scale, mini_batch_size=128, temporary_device='cpu'):

        dataset = TensorDataset(mu_nb*scale, theta, pi)
        dataloader = DataLoader(dataset,shuffle=False, batch_size=mini_batch_size)

        counts_list = []

        with torch.no_grad():
            for minibatch in dataloader:
                zinb = ZINB(mu=minibatch[0], theta=minibatch[1], pi=minibatch[2])
                counts_list.append(zinb.sample())

        counts = torch.cat(counts_list, dim=0)

        return counts

    def remove_batch(self, z, additional_removal=None,  replace_values=0, mini_batch_size=None):

        if additional_removal is None:
            additional_removal = []

        removal = ['batch_effect'] + additional_removal

        if isinstance(replace_values, int) or isinstance(replace_values, float) or replace_values=='mean' or replace_values=='median':
            replace_values = [replace_values] * (len(removal))

        elif isinstance(replace_values, list):
            if len(removal) != len(replace_values):
                raise ValueError("Length of additional_removal must be length of replace_values - 1" )

        else:
            raise ValueError("'replace_values' must be numeric, 'median', 'mean' or list of these.")

        z_corrected = z.clone()


        for i,colname in enumerate(removal):

            if replace_values[i] == 'mean':
                z_corrected[:, self.dim_dict[colname]] = z_corrected[:, self.dim_dict[colname]].mean(dim=0)

            elif replace_values[i] == 'median':
                z_corrected[:, self.dim_dict[colname]] = z_corrected[:, self.dim_dict[colname]].median(dim=0).values

            else:
                z_corrected[:, self.dim_dict[colname]] = replace_values[i]

        return z_corrected

    def _generate_file_prefix(self):

        now = datetime.datetime.now()
        date = now.date().strftime('%Y%m%d')
        file_prefix = 'Kanade_{}'.format(date)

        return file_prefix

    def save(self):

        """インスタンスの設定とモデルを保存"""
        save_dict = self.__dict__.copy()
        save_dict.pop("model")  # モデルは別途保存
        save_dict.pop("train_loader", None)  # DataLoaderは保存できないので除外
        save_dict.pop("test_loader", None)

        file_name = self.model_file + '_setting.pkl'

        with open(file_name, "wb") as f:
            pickle.dump(save_dict, f)

        torch.save(self.model.state_dict(), self.model_file)

        print(f"Saved settings to {file_name}")

    def __str__(self):

        return 'Kanade object'

    def __repr__(self):

        return_text_input = f'input_dim: {self.input_dim}'

        return_text_supervised = f'Predictors:'
        for col,d,n,t in zip(self.supervised_columns, self.supervised_dim, self.dim_supervised_target, self.predictor_type):

            if t in ['c', 'C']:
                t = 'classifier'

            elif t in ['r', 'R']:
                t = 'regressor'

            return_text_supervised = return_text_supervised+f'\n\t{col}: {d} latent variables, {n} targets, {t}'

        return_text_unsupervised = f'unsupervised_dim: {self.unsupervised_dim}'

        return_text_batch = f'batch_column: {self.batch_column}\nbatch_predictor: {self.batch_dim} latent variables, {self.num_batch} targets'

        return_text_filter = f'filter_sizes: {self.filter_sizes}'

        return_text_kwargs_vae = f'kwargs_vae: {self.kwargs_vae}'

        return_text_kwargs_loss_fn = f'kwargs_loss_fn: {self.kwargs_loss_fn}'

        if self.do_train_test_split:
            test_ratio = f'test_ratio: {self.test_ratio}, seed {self.split_seed}'

        else:
            test_ratio = f'test_ratio: Without train/test splitting'

        return_text_train_test_split = f'train_test_split: {test_ratio}'

        return_text_other_training = f'mini batch size: {self.mini_batch_size}\nlearning_seed: {self.learning_seed}'

        return_text_device = f'device: {self.device}'

        return_text_legacy = f'legacy: {self.legacy}'

        if self.log_file is None:
            log_file = None
        else:
            log_file = self.log_file + '_*.log'

        return_text_files = f'model_file: {self.model_file}\nlog_file: {log_file}'

        return_text = '\n'.join(['< Kanade object',
            return_text_input,
            return_text_supervised,
            return_text_unsupervised,
            return_text_batch,
            return_text_filter,
            return_text_kwargs_vae,
            return_text_kwargs_loss_fn,
            return_text_train_test_split,
            return_text_other_training,
            return_text_device,
            return_text_legacy,
            return_text_files,
            '>']
        )

        return return_text







    @property
    def _hisaishi(self):

        print('𝄢 The package name "Kanade" is named from Kanade Hisaishi. She is a charactor of "Sound! Euphonium", a novel series by Ayano Takeda.')


def load_kanade(model_file, setting_file=None, device=None):

    if setting_file is None:
        setting_file = model_file + '_setting.pkl'

    with open(setting_file, "rb") as f:
        save_dict = pickle.load(f)

    knd = Kanade.__new__(Kanade)  # __init__ を呼ばずにインスタンスを作成
    knd.__dict__.update(save_dict)

    knd.legacy = getattr(knd, "legacy", True)
    #互換性。旧バージョンはlagacy属性がなく、legacy手法
    #読み込んでlegacy属性がなければTrueとする

    knd.model = knd.generate_vae(knd.input_dim,
        knd.supervised_dim,
         knd.dim_supervised_target,
         knd.unsupervised_dim,
         knd.batch_dim,
         knd.num_batch,
         knd.predictor_type,
         knd.filter_sizes,
         knd.legacy,
         knd.kwargs_vae,
         knd.do_parallel)

    if not device is None: knd.device = device

    knd.model.load_state_dict(torch.load(model_file, map_location=knd.device))
    knd.model.to(knd.device)
    knd.model.eval()

    return knd
