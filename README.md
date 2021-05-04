## Efficient Robustness Certificates for Discrete Data 

<p align="center">
<img src="https://www.in.tum.de/fileadmin/w00bws/daml/sparse_smoothing/sparse_smoothing.png" width="500">

Reference implementation of the certificates proposed in the paper:

["Efficient Robustness Certificates for Discrete Data: Sparsity-Aware Randomized Smoothing for Graphs, Images and More"](https://proceedings.icml.cc/static/paper_files/icml/2020/6890-Paper.pdf)

Aleksandar Bojchevski, Johannes Klicpera, and Stephan Günnemann, ICML 2020.

## Example
The notebook demo.ipynb shows an example of how to use our binary certificate for a pretrained GCN model. You can use `scripts/train_and_cert.py` to train and certify a model from scratch on a cluster using [SEML](https://github.com/TUM-DAML/seml).

## Cite
Please cite our paper if you use this code in your own work:

```
@inproceedings{bojchevski_sparsesmoothing_2020,
title = {Efficient Robustness Certificates for Discrete Data: Sparsity-Aware Randomized Smoothing for Graphs, Images and More},
author = {Bojchevski, Aleksandar and Klicpera, Johannes and G{\"u}nnemann, Stephan},
booktitle={Proceedings of Machine Learning and Systems 2020},
pages = {11647--11657},
year = {2020}
}
```
