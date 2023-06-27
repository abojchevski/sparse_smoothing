## Efficient Robustness Certificates for Discrete Data 

<p align="center">
<img src="https://www.cs.cit.tum.de/fileadmin/_processed_/f/6/csm_sparse_smoothing_c062042b99.png" width="500">

Reference implementation of the certificates proposed in the paper:

["Efficient Robustness Certificates for Discrete Data: Sparsity-Aware Randomized Smoothing for Graphs, Images and More"](https://arxiv.org/abs/2008.12952)

Aleksandar Bojchevski, Johannes Gasteiger, and Stephan Günnemann, ICML 2020.

## Example
The notebook demo.ipynb shows an example of how to use our binary certificate for a pretrained GCN model. You can use `scripts/train_and_cert.py` to train and certify a model from scratch on a cluster using [SEML](https://github.com/TUM-DAML/seml).

## Cite
Please cite our paper if you use this code in your own work:

```
@inproceedings{bojchevski_sparsesmoothing_2020,
title = {Efficient Robustness Certificates for Discrete Data: Sparsity-Aware Randomized Smoothing for Graphs, Images and More},
author = {Bojchevski, Aleksandar and Gasteiger, Johannes and G{\"u}nnemann, Stephan},
booktitle = {Proceedings of the 37th International Conference on Machine Learning},
pages = {1003--1013},
year = {2020}
}
```
