# CeSpGRN: Inferring cell specific GRN using single-cell gene expression data 

[Zhang's Lab](https://xiuweizhang.wordpress.com), Georgia Institute of Technology

Developed by Ziqi Zhang, Jongseok Han

**Preprint:** available soon

## Description
CeSpGRN is a package that is able to infer cell specific GRN using single-cell gene expression data

* `src` stores the inference algorithms.
* `test` stores the `scripts` (testing scripts) and `results` (testing results are directly generated by the scripts, too large to be pushed onto github).
  * `scripts_GGM`: testing script for the GGM data
  * `scripts_softODE`: testing script for the softODE data
  * `scripts_THP-1`: testing script for the THP-1 data
* `simulator` stores the simulation code:
  * `GGM`: simulator for GGM data
  * `soft_boolODE`: simulator for the softODE data
* `data` stores the real and simulated data (available upon requests)


## Dependency
```
(required)
pytorch >= 1.15.0 
numpy >= 1.19.5
scipy >= 1.7.1
networkx >= 2.5
sklearn >= 0.24.2

(optional)
matplotlib >= 3.4.3
statsmodels >= 0.12.2
```

## Usage
* Load in the count matrix as a numpy `ndarray`, the matrix should be of the shape `(ncells, ngenes)`. e.g.
  ```python
  import sys, os
  sys.path.append('./src/')
  import numpy as np 

  # load CeSpGRN
  import g_admm as CeSpGRN
  import kernel

  # read in count matrix
  counts = np.load("counts.npy")
  ```
* Set the hyper-parameter including: bandwidth, neighborhoodsize, and lambda. e.g.
  ```python
  # smaller bandwidth means that GRN of cells are more heterogeneous.
  bandwidth = 1
  # number of neighbor being considered when calculating the covariance matrix.
  n_neigh = 30
  # sparsity regulatorization, larger lamb means sparser result.
  lamb = 0.1
  ```
* Calculate the kernel function, and covariance matrix for each cell, e.g.
  ```python
  # calculate PCA of count matrix
  from sklearn.decompose import PCA
  pca_op = PCA(n_components = 10)
  X_pca = pca_op.fit_transform(counts)

  # using X_pca to calculate the kernel function
  K, K_trun = kernel.calc_kernel_neigh(X_pca, k = 5, bandwidth = bandwidth, truncate = True, truncate_param = n_neigh)

  # estimate covariance matrix, output is empir_cov of the shape (ncells, ngenes, ngenes)
  empir_cov = CeSpGRN.est_cov(X = counts, K_trun = K_trun, weighted_kt = True)
  ```
* Estimating cell-specific GRN, e.g.
  ```python
  # estimate cell-specific GRNs, thetas of the shape (ncells, ngenes, ngenes)
  cespgrn = CeSpGRN.G_admm_minibatch(X=counts[:, None, :], K=K, pre_cov=empir_cov, batchsize = 120)
  thetas = cespgrn.train(max_iters=max_iters, n_intervals=100, lamb=lamb)
  ```

An example run is shown in `demo.py`.
