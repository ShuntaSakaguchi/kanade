# Kanade API Reference

## Kanade

```python
kanade.Kanade(
adata,
supervised_columns,
batch_column,
dim_supervised_target,
num_batch,
supervised_dim,
predictor_type = None,
unsupervised_dim=0,
batch_dim=10,
filter_sizes=None,
do_train_test_split=True,
test_ratio=0.1,
split_seed=0,
learning_seed=0,
mini_batch_size=128,
kwargs_vae = None,
kwargs_loss_fn=None,
do_parallel=False,
*args,
**kwargs)
```
### Description
Kanade object class

### Parameters

- **adata** : `anndata.AnnData`  
  Dataset to correct. adata.X should have a raw counts matrix with `scipy.sparse.csr_matrix` format, and adata.obs must contain metadata to train predictors.

- **supervised_columns** : list of `str`  
  List of metadata labels (other than batch) to be learned by predictors. Corresponding latent variables are created to preserve signals across conditions.

- **batch_column** : `str`  
  Metadata label of batches. Corresponding latent variables are created and be corrected in batch correction process

- **dim_supervised_target** : list of `int`  
  The number of output values for each predictor. Specify the values in the same order as the labels listed in **supervised_columns**. For categorical labels, provide the number of categories; for continuous labels, use 1.　The length of list must be the same as **supervised_columns**.

- **num_batch** : `int`  
  The number of batches, it must be the same number of categories in the metadata in **batch_column**.

- **supervised_dim**  : `int` or list of `int`  
  The numbers of latent variables corresponding to labels listed in **supervised_columns**. If an `int` is given, the same number of latent variables is assigned to each label. If a `list` is given, specify the number of latent variables for each label in the same order as in **supervised_columns**. In this case, the length of the list must match the length of **supervised_columns**.

- **predictor_type** : (optional) `str` or list of `str`, default = `None`  
  Types of predictors. Use "r" for regressors (continuous labels) and "c" for classifiers (categorical labels). If `None`, all predictors are treated as classifiers. If a single `str` value ("r" or "c") is provided, all predictors are assigned that type. If a `list` is provided, the predictor types must be specified in the same order as the labels given in supervised_columns, using "r" or "c" for each entry. In this case, the length of the list must be the same as the length of supervised_columns.

- **unsupervised_dim** : (optional) `int`, default = `0`  
  The number of additional latent variables without connection of predictors.

- **batch_dim** : (optional) `int`, default = `10`  
  The number of latent variables corresponding to batches.

- **filter_sizes** : (optional) list of `int`, default = `None`  
  The number of nodes in the encoder, specified as a list ordered from the input layer to the latent layer.
  The decoder uses the reverse order of this list. If `None`, it is set to [2000, 400, 200].

- **do_train_test_split** : (optional) `bool`, default = `True`  
  If True, the dataset is split into training and test sets, and only the training set is used for model training. During training, the loss is also computed and recorded on the test set separately from the training data.

- **test_ratio** : (optional) `float`, default = `0.1`  
  Proportion of the dataset to use as the test set. Ignored if **do_train_test_split** = `False`.

- **split_seed** : (optional) `int`, default = `0`  
  Random seed to split the dataset into training and test sets. Ignored if **do_train_test_split** = `False`.

- **learning_seed** : (optional) `int`, default = `0`  
  Random seed for the training process.

- **mini_batch_size** : (optional) `int`, default = `128`  
  Mini-batch size for training process.

- **kwargs_vae** : (optional) `dict`, default = `None`  
  Additional arguments for `kanade_module.VAE_kanade` class.

- **kwargs_loss_fn** : (optional) `dict`, default = `None`  
  Additional arguments for `kanade_module.VAELoss_kanade` class. **Weights of loss functions can be set here.**

- **do_parallel** : (optional) `bool`, default = `True`  
  If True, enables model parallelization via torch.nn.DataParallel. Requires multiple GPUs.
  <br>
  <br>
### Methods

```python
kanade.Kanade.training(
  self,
  model_file=None,
  log_file_prefix=None,
  optimizer = None,
  logging=True,
  n_epochs=50,
  learning_rate=0.001)
```
#### Description

The method to train the model.

#### Parameters

- **model_file** : (optional) `str`, default = `None`  
  File name to save the weights. Extension of file is `.pth`. Setting of the Kanade object is saved to the file named "**model_file** + '_setting.pkl'". If `None`, "Kanade\_{date}" is set.

- **log_file_prefix** : (optional) `str`, default = `None`
  Prefix of the name of log files. The log of loss for the training data is wrote to the file named  "**log_file_prefix**+'_train.log'" and for the test data to "**log_file_prefix**+'_test.log'". If `None`, "Kanade\_{date}" is set. It is ignored if **logging** == `False`.

- **optimizer** : (optional) `torch.optim.Optimizer`, default = `None`  
  Optimizer object for the training. If `None`, `torch.optim.Adam` is used.

- **logging** : (optional) `bool`, default = `True`  
  If `True`, values of loss functions are recorded in the training.

- **n_epochs** : (optional) `int`, default = `50`  
  The number of training epochs.

- **learning_rate** : (optional) `float`, default = `0.001`  
  Learning rate for `torch.optim.Adam`. Ignored if **optimizer** is set.

#### Returns

- **self** : `kanade.Kanade`  
  Trained model.

<br>
<br>

```python
kanade.Kanade.projection(
  self,
  X,
  mini_batch_size=1024,
  temporary_device='cpu',
  return_mu_log_var=False)
```
#### Description

The method to project data into the latent space.

#### Parameters

- **X** : `scipy.sparse.csr_matrix`  
  Count matrix. size : `n_obs` x `n_features`

- **mini_batch_size** : (optional) `int`, default = `1024`  
  Mini-batch size for projection process.

- **temporary_device** : (optional) `torch.device`, default = `cpu`  
  Device in which projected data is put.

- **return_mu_log_var** : (optional) `bool`, default = `False`  
  If `True`, parameters (mean and log-variance) of posterior distributions of latent variables are returned with latent variables.

#### Returns

- **z** : `torch.Tensor`  
  Calculated latent variables. Shape: n_obs × n_latent_variables. The variables are arranged in the following order:  (1) a single scale-factor variable,  (2) latent variables associated with supervised_columns and (3) batch-related latent variables. The numbers of variables in (2) and (3) are determined by the values specified in the constructor.

- **mu** : `torch.Tensor`  
  Means of posterior distributions of latent variables. Shape is the same as **z**. It is returned only if **return_mu_log_var** is `True`.

- **log_var** : `torch.Tensor`  
  log-variances of posterior distributions of latent variables. Shape is the same as **z**. It is returned only if **return_mu_log_var** is `True`.

  <br>
  <br>

  ```python
  kanade.Kanade.decode(
    self,
    z,
    mini_batch_size=1024,
    temporary_device='cpu'
    )
  ```
#### Description

Decode latent variables and calculate parameters of ZINB.

#### Parameters

- **z** : `torch.Tensor`
  Latent variables to use to calculate parameters of ZINB.

- **mini_batch_size** : (optional) `int`, default = `1024`
  Mini-batch size for decoding process.

- **temporary_device** : (optional) `torch.device`, default = `cpu`  
  Device in which parameters are put.

#### Returns

- **mu_nb** : `torch.Tensor`  
  Scaled mean parameters of ZINB. size : `n_obs` x `n_vars`

- **theta** : `torch.Tensor`  
  xxx parameters of ZINB. Shared across all features. size : `n_obs` x `1`.

- **pi** : `torch.Tensor`  
  Dropout rate parameters of ZINB. size : `n_obs` x `n_vars`

- **scale**: `torch.Tensor`  
  Scale factors of each cells. size : `n_obs` x `1`

  <br>
  <br>

```python
kanade.Kanade.resample(
  self,
  mu_nb,
  theta,
  pi,
  scale,
  mini_batch_size=128,
  temporary_device='cpu')
```

#### Description

Regenerate the count matrix from a ZINB distribution using the given parameters.

#### Parameters

- **mu_nb** : `torch.Tensor`  
  Scaled mean parameters of ZINB. size : `n_obs` x `n_vars`

- **theta** : `torch.Tensor`  
  xxx parameters of ZINB. Shared across all features. size : `n_obs` x `1`.

- **pi** : `torch.Tensor`  
  Dropout rate parameters of ZINB. size : `n_obs` x `n_vars`

- **scale**: `torch.Tensor`  
  Scale factors of each cells. size : `n_obs` x `1`

- **mini_batch_size** : (optional) `int`, default = `128`  
  Mini-batch size for resampling process.

- **temporary_device** : (optional) `torch.device`, default = `cpu`  
  Device in which reconstructed data is put.

#### Returns

- **counts** : `torch.Tensor`  
  Reconstructed counts. size : `n_obs` x `n_vars`

  <br>
  <br>

```python
kanade.Kanade.remove_batch(
  self,
  z,
  additional_removal=None,  
  replace_values=0)
```

#### Description
Correct latent variables.

#### Parameters

- **z** : `torch.Tensor`  
  Calculated latent variables. Shape: n_obs × n_latent_variables. The variables must be arranged in the following order:  (1) a single scale-factor variable,  (2) latent variables associated with supervised_columns and (3) batch-related latent variables. The numbers of variables in (2) and (3) are determined by the values specified in the constructor.

- **additional_removal** : (optional) list of `str`  
  Metadata label names corresponding to latent variables to be corrected in addition to batch. Must be selected from the labels specified in **supervised_columns** parameter of the constructor. If `None`, only the batch-related latent variables are corrected.

- **replace_values** : (optional) `numeric`, `str`, or list of them  
  Specifies the value used to replace latent variables during correction. If a `numeric` value is given, latent variables are replaced with that value. If `'mean'` is given, they are replaced with the mean value; if `'median'`, with the median value. A list of such values can also be provided. In this case, the first element corresponds to the batch latent variables, and the subsequent elements correspond to the labels specified in **additional_removal**, in the same order. When a list is given, its length must be equal to len(**additional_removal**) + 1.

#### Returns

- **z_corrected** : `torch.Tensor`  
  Corrected latent variables.

  <br>
  <br>

```python
kanade.Kanade.save(
  self):
```

#### Description
Save trained Kanade model. File name is set in the `training` method.

<br>
<br>

## load_kanade

```python
kanade.load_kanade(
  model_file,
  setting_file=None,
  device=None):
```

### Description
Function to load trained Kanade object from files.


### Parameters

- **model_file** : `str`
  File name prefix in which data is saved.  Weights of models are loaded from **model_file** + `'.pth'`. Other object setting is load from **model_file** + `'_setting.pkl'` if **setting_file** parameter is None.

- **setting_file** : (optional, Not recommended) `str`  
  File name of object setting other than weights.  

- **device** : (optional, Not recommended) `torch.device`
  Device to put weights.
