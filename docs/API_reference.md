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

### Parameters

- **adata** : `anndata.AnnData`  
  Dataset to correct. adata.X should have a raw counts matrix, and adata.obs must contain metadata to train predictors.

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

- **learning_seed** : (optional) 'int', default = `0`  
  Random seed for the training process.

- **mini_batch_size** : (optional) `int`, default = `128`  
  Mini-batch size for training process.

- **kwargs_vae** : (optional) `dict`, default = `None`  
  Additional arguments for `kanade_module.VAE_kanade` class.

- **kwargs_loss_fn** : (optional) `dict`, default = `None`  
  Additional arguments for `kanade_module.VAELoss_kanade` class. **Weights of loss functions can be set here.**

- **do_parallel** : (optional) `bool`, default = `True`  
  If True, enables model parallelization via torch.nn.DataParallel. Requires multiple GPUs.

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

#### Parameters

- **model_file** : `str`, default = `None`  
