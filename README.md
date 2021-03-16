**Deep k-Means: Jointly Clustering with k-Means and Learning Representations**
======

## __Introduction__

This repository provides the source code for the models and baselines described in *Seed-guided Deep Document Clustering* by Maziar Moradi Fard, Thibaut Thonet, and Eric Gaussier. The implementation is based on Python and Tensorflow. More details about this work can be found in the original paper, which is available at https://link.springer.com/chapter/10.1007/978-3-030-45439-5_1 .

**Abstract:** Different users may be interested in different clustering views underlying a given collection (e.g., topic and writing style in documents). Enabling them to provide constraints reflecting their needs can then help obtain tailored clustering results. For document clustering, constraints can be provided in the form of seed words, each cluster being characterized by a small set of words. This seed-guided constrained document clustering problem was recently addressed through topic modeling approaches. In this paper, we jointly learn deep representations and bias the clustering results through the seed words, leading to a Seed-guided Deep Document Clustering approach. Its effectiveness is demonstrated on five public datasets.

If you found this implementation useful, please consider citing us:
Moradi Fard, M., Thonet, T., & Gaussier, E. (2018). **[Seed-guided Deep Document Clustering](https://link.springer.com/chapter/10.1007/978-3-030-45439-5_19)**.

Feel free to contact us if you discover any bugs in the code or if you have any questions.

## __Content__

The repository contains the following files:
* The python scripts used to run the different models and baselines: **cdkm_ce.py**, **cdkm_cc.py**. 
* The python scripts **compgraph_ce(cc).py** containing the Tensorflow computation graph of the different models.
* The python scripts describing the specifications of the datasets (used for dataset loading, preprocessing, and dataset-specific parameter setting): **complete_reuters_masked_specs_w2v** which contains specifications of the Reuters dataset. The specification of the other datasets will be available soon.
* The python script **utils.py**, which defines basic functions.
* The directory **constraints** containing the constraints obtained from different datasets.
* The file **LICENCE.txt** describing the licence of our code.
* The file **README.md**, which is the current file.

## __How to run the code__

Deep k-Means (DKM) is run using the following command:
```cdkm_ce(cc).py [-h] -d <string> [-v] [-p] [-a] [-s] [-c] [-l <double>] [-e <int>] [-f <int>] [-b <int>]```

The meaning of each argument is detailed below:
* ``-h``, ``--help``: Show usage.
* ``-d <string>``, ``--dataset <string>``: Dataset on which DKM will be run (one of USPS, MNIST, 20NEWS, RCV1).
* ``-v``, ``--validation``: Split data into validation and test sets.
* ``-p``, ``--pretrain``: Pretrain the autoencoder and cluster representatives.
* ``-a``, ``--annealing``: Use an annealing scheme for the values of alpha (otherwise a constant is used).
* ``-s``, ``--seeded``: Use a fixed seed, different for each run.
* ``-c``, ``--cpu``: Force the program to run on CPU.
* ``-l0 <double>``, ``--lambda <double>``: Value of the hyperparameter weighing the clustering loss against the reconstruction loss and constrained loss. Default value: 1.0.
* ``-l1 <double>``, ``--lambda <double>``: Value of the hyperparameter weighing the constrained loss against the reconstruction loss and clustering loss. Default value: 1.0.
* ``-e <int>``, ``--p_epochs <int>``: Number of pretraining epochs. Default value: 50.
* ``-f <int>``, ``--f_epochs <int>``: Number of fine-tuning epochs per alpha value. Default value: 5.
* ``-b <int>``, ``--batch_size <int>``: Size of the minibatches used by the optimizer. Default value: 256.
