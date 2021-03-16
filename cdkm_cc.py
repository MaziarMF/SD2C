#!/usr/bin/env python3

__author__ = "Thibaut Thonet, Maziar Moradi Fard"
__license__ = "GPL"

import os
import math
import numpy as np
import argparse
import tensorflow as tf
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
from utils import cluster_acc
from compgraph_cc import CDKMCombined


parser = argparse.ArgumentParser(description="Deep k-means algorithm")
parser.add_argument("-d", "--dataset", type=str.upper,
                    help="Dataset on which DKM will be run (one of USPS, MNIST, 20NEWS, RCV1)", required=True)
parser.add_argument("-v", "--validation", help="Split data into validation and test sets", action='store_true')
parser.add_argument("-p", "--pretrain", help="Pretrain the autoencoder and cluster representatives",
                    action='store_true')
parser.add_argument("-a", "--annealing",
                    help="Use an annealing scheme for the values of alpha (otherwise a constant is used)",
                    action='store_true')
parser.add_argument("-s", "--seeded", help="Use a fixed seed, different for each run", action='store_true')
parser.add_argument("-c", "--cpu", help="Force the program to run on CPU", action='store_true')
parser.add_argument("-l0", "--lambda0", type=float, default=1, dest="lambda0_",
                    help="Value of the hyperparameter weighing the clustering loss")
parser.add_argument("-l1", "--lambda1", type=float, default=1, dest="lambda1_",
                    help="Value of the hyperparameter weighing the constraint loss")
parser.add_argument("-l2", "--lambda2", type=float, default=1, dest="lambda2_",
                    help="Value of the hyperparameter weighing the constraint loss cdc part")
parser.add_argument("-l3", "--lambda3", type=float, default=1, dest="lambda3_",
                    help="Value of the hyperparameter weighing the constraint loss cdkm part")
parser.add_argument("-l", "--lambda", type=float, default=1, dest="lambda_",
                    help="Value of the hyperparameter weighing the alternation between cdkm and cdc")
parser.add_argument("-e", "--p_epochs", type=int, default=50, help="Number of pretraining epochs")
parser.add_argument("-f", "--f_epochs", type=int, default=5, help="Number of fine-tuning epochs per alpha value")
parser.add_argument("-b", "--batch_size", type=int, default=256, help="Size of the minibatches used by the optimizer")

args = parser.parse_args()

# Dataset setting from arguments

if args.dataset == "20NEWS":
    import _20news_masked_specs_w2v as specs
    print("specs file corresponding to masked version is loaded...")
    data = specs.data
elif args.dataset == "DBPEDIA":
    import dbpedia_masked_specs_w2v as specs
    print("specs file corresponding to masked version is loaded...")
    data = specs.data
elif args.dataset == "YAHOO":
    import yahoo_masked_specs_w2v as specs
    data = specs.data
elif args.dataset == "REUTERS":
    import reuters_masked_specs_w2v as specs
    data = specs.data
elif args.dataset == "AGNEWS":
    import agnews_masked_specs_w2v as specs
    data = specs.data
else:
    parser.error("Unknown dataset!")
    exit()

# Parameter setting from arguments
n_pretrain_epochs = args.p_epochs
n_finetuning_epochs = args.f_epochs
val_lambda0 = args.lambda0_
val_lambda1 = args.lambda1_
val_lambda2 = args.lambda2_
val_lambda3 = args.lambda3_
val_lambda = args.lambda_


batch_size = args.batch_size # Size of the mini-batches used in the stochastic optimizer
n_batches = int(math.ceil(specs.n_samples / batch_size)) # Number of mini-batches
validation = args.validation # Specify if data should be split into validation and test sets
pretrain = args.pretrain # Specify if DKM's autoencoder should be pretrained
annealing = args.annealing # Specify if annealing should be used
seeded = args.seeded # Specify if runs are seeded


data_sec, data_masked = specs.data_second, specs.data_masked
from utils import next_batch_masked as next_batch



print("Hyperparameters...")
print("lambda0 =", val_lambda0)
print("lambda1 =", val_lambda1)
print("lambda2 =", val_lambda2)
print("lambda3 =", val_lambda3)
print("lambda =", val_lambda)

# Define the alpha scheme depending on if the approach includes annealing/pretraining
if annealing and not pretrain:
    constant_value = 1  # specs.embedding_size # Used to modify the range of the alpha scheme
    max_n = 40  # Number of alpha values to consider
    alphas = np.zeros(max_n, dtype=float)
    alphas[0] = 0.1
    for i in range(1, max_n):
        alphas[i] = (2 ** (1 / (np.log(i + 1)) ** 2)) * alphas[i - 1]
    alphas = alphas / constant_value
elif not annealing and pretrain:
    constant_value = 1  # specs.embedding_size # Used to modify the range of the alpha scheme
    max_n = 20  # Number of alpha values to consider (constant values are used here)
    alphas = 1000*np.ones(max_n, dtype=float) # alpha is constant
    alphas = alphas / constant_value
else:
    parser.error("Run with either annealing (-a) or pretraining (-p), but not both.")
    exit()

# Select only the labels which are to be used in the evaluation (disjoint for validation and test)
if validation:
    validation_target = np.asarray([specs.target[i] for i in specs.validation_indices])
    test_target = np.asarray([specs.target[i] for i in specs.test_indices])
else:
    target = specs.target


# Hardware specifications
if args.cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Run on CPU instead of GPU if batch_size is small
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
config = tf.ConfigProto(gpu_options=gpu_options)

# Definition of the randomly-drawn (0-10000) seeds to be used for each run
seeds = [8905, 9129, 291, 4012, 1256, 6819, 4678, 6971, 1362, 575]

if validation:
    list_validation_acc = []
    list_validation_ari = []
    list_validation_nmi = []
    list_test_acc = []
    list_test_ari = []
    list_test_nmi = []
else:
    list_acc = []
    list_ari = []
    list_nmi = []

n_runs = 10
for run in range(n_runs):
    # Use a fixed seed for this run, as defined in the seed list
    if seeded:
        tf.reset_default_graph()
        tf.set_random_seed(seeds[run])
        np.random.seed(seeds[run])

    print("Run", run)
    # Define the computation graph for DKM
    cg = CDKMCombined([specs.dimensions, specs.activations, specs.names], specs.keywords, specs.n_clusters, 
                       val_lambda0, val_lambda1, val_lambda2, val_lambda3, val_lambda)

    # Run the computation graph
    with tf.Session(config=config) as sess:
        # Initialization
        init = tf.global_variables_initializer()
        sess.run(init)

        # Variables to save tensor content
        distances = np.zeros((specs.n_clusters, specs.n_samples))

        # Pretrain if specified
        if pretrain:
            print("Starting autoencoder pretraining...")

            # Variables to save pretraining tensor content
            embeddings = np.zeros((specs.n_samples, specs.embedding_size), dtype=float)

            # First, pretrain the autoencoder
            ## Loop over epochs
            for epoch in range(n_pretrain_epochs):
                print("Pretraining step: epoch {}".format(epoch))

                # Loop over the samples
                for _ in range(n_batches):
                    # Fetch a random data batch of the specified size
                    indices, batch_data, sec_data, masked_data = next_batch(batch_size, data, data_sec, data_masked)
                    _, embedding_, ae_loss_, keywords_embedding_ = sess.run((cg.pretrain_op, cg.embedding, cg.ae_loss, cg.keywords_embedding),
                                                                            
                               feed_dict={cg.input: batch_data, cg.data_masked: masked_data, cg.data_second: sec_data})
                    # Save the embeddings for batch samples
                    for j in range(len(indices)):
                        embeddings[indices[j], :] = embedding_[j, :]

                    #print("ae_loss_:", float(ae_loss_))

            # Second, run k-means++ on the pretrained embeddings
            print("Running k-means on the learned embeddings...")
            kmeans_model = KMeans(n_clusters=specs.n_clusters, init="k-means++").fit(embeddings)

            if validation:
                # Split the cluster assignments into validation and test ones
                validation_cluster_assign = np.asarray([kmeans_model.labels_[i] for i in specs.validation_indices])
                test_cluster_assign = np.asarray([kmeans_model.labels_[i] for i in specs.test_indices])

                # Evaluate the clustering validation performance using the ground-truth labels
                validation_acc = cluster_acc(validation_target, validation_cluster_assign)
                print("Validation ACC", validation_acc)
                validation_ari = adjusted_rand_score(validation_target, validation_cluster_assign)
                print("Validation ARI", validation_ari)
                validation_nmi = normalized_mutual_info_score(validation_target, validation_cluster_assign)
                print("Validation NMI", validation_nmi)

                # Evaluate the clustering test performance using the ground-truth labels
                test_acc = cluster_acc(test_target, test_cluster_assign)
                print("Test ACC", test_acc)
                test_ari = adjusted_rand_score(test_target, test_cluster_assign)
                print("Test ARI", test_ari)
                test_nmi = normalized_mutual_info_score(test_target, test_cluster_assign)
                print("Test NMI", test_nmi)
            else:
                # Evaluate the clustering performance using the ground-truth labels
                acc = cluster_acc(target, kmeans_model.labels_)
                print("ACC", acc)
                ari = adjusted_rand_score(target, kmeans_model.labels_)
                print("ARI", ari)
                nmi = normalized_mutual_info_score(target, kmeans_model.labels_)
                print("NMI", nmi)
              
            #initializing cluster centers with keywords embedding
            sess.run(tf.assign(cg.cluster_rep, cg.keywords_embedding))


        # Train the full CDKM model
        if (len(alphas) > 0):
            print("Starting CDKM training...")
        ## Loop over alpha (inverse temperature), from small to large values
        for k in range(len(alphas)):
            print("Training step: alpha[{}]: {}".format(k, alphas[k]))

            # Loop over epochs per alpha
            for _ in range(n_finetuning_epochs):
                # Loop over the samples
                for _ in range(n_batches):
                    indices, batch_data, sec_data, masked_data = next_batch(batch_size, data, data_sec, data_masked)

                    #print(tf.trainable_variables())
                    #current_batch_size = np.shape(data_batch)[0] # Can be different from batch_size for unequal splits

                    # Run the computation graph on the data batch
                    _, loss_, stack_dist_, ae_loss_, clustering_loss_ , constrained_loss_cdc_, constrained_loss_ckm_ =\
                    sess.run((cg.train_op, cg.loss, cg.stack_dist, cg.ae_loss, cg.clustering_loss, cg.constrained_loss_cdc, cg.constrained_loss_cdkm),
                                 feed_dict={cg.input: batch_data, cg.alpha: alphas[k], cg.data_masked: masked_data, cg.data_second: sec_data})

                    # Save the distances for batch samples
                    for j in range(len(indices)):
                        distances[:, indices[j]] = stack_dist_[:, j]

                

            # Evaluate the clustering performance every print_val alpha and for last alpha
            print_val = 1
            if k % print_val == 0 or k == max_n - 1:
                print("loss:", loss_)
                print("ae loss:", ae_loss_)
                print("clustering loss:", clustering_loss_)

                # Infer cluster assignments for all samples
                cluster_assign = np.zeros((specs.n_samples), dtype=float)
                for i in range(specs.n_samples):
                    index_closest_cluster = np.argmin(distances[:, i])
                    cluster_assign[i] = index_closest_cluster
                cluster_assign = cluster_assign.astype(np.int64)

                if validation:
                    validation_cluster_assign = np.asarray([cluster_assign[i] for i in specs.validation_indices])
                    test_cluster_assign = np.asarray([cluster_assign[i] for i in specs.test_indices])

                    # Evaluate the clustering validation performance using the ground-truth labels
                    validation_acc = cluster_acc(validation_target, validation_cluster_assign)
                    print("Validation ACC", validation_acc)
                    validation_ari = adjusted_rand_score(validation_target, validation_cluster_assign)
                    print("Validation ARI", validation_ari)
                    validation_nmi = normalized_mutual_info_score(validation_target, validation_cluster_assign)
                    print("Validation NMI", validation_nmi)

                    # Evaluate the clustering test performance using the ground-truth labels
                    test_acc = cluster_acc(test_target, test_cluster_assign)
                    print("Test ACC", test_acc)
                    test_ari = adjusted_rand_score(test_target, test_cluster_assign)
                    print("Test ARI", test_ari)
                    test_nmi = normalized_mutual_info_score(test_target, test_cluster_assign)
                    print("Test NMI", test_nmi)
                else:
                    # Evaluate the clustering performance using the ground-truth labels
                    acc = cluster_acc(target, cluster_assign)
                    print("ACC", acc)
                    ari = adjusted_rand_score(target, cluster_assign)
                    print("ARI", ari)
                    nmi = normalized_mutual_info_score(target, cluster_assign)
                    print("NMI", nmi)

        # Record the clustering performance for the run
        if validation:
            list_validation_acc.append(validation_acc)
            list_validation_ari.append(validation_ari)
            list_validation_nmi.append(validation_nmi)
            list_test_acc.append(test_acc)
            list_test_ari.append(test_ari)
            list_test_nmi.append(test_nmi)
        else:
            list_acc.append(acc)
            list_ari.append(ari)
            list_nmi.append(nmi)

if validation:
    list_validation_acc = np.array(list_validation_acc)
    print("Average validation ACC: {:.3f} +/- {:.3f}".format(np.mean(list_validation_acc), np.std(list_validation_acc)))
    list_validation_ari = np.array(list_validation_ari)
    print("Average validation ARI: {:.3f} +/- {:.3f}".format(np.mean(list_validation_ari), np.std(list_validation_ari)))
    list_validation_nmi = np.array(list_validation_nmi)
    print("Average validation NMI: {:.3f} +/- {:.3f}".format(np.mean(list_validation_nmi), np.std(list_validation_nmi)))

    list_test_acc = np.array(list_test_acc)
    print("Average test ACC: {:.3f} +/- {:.3f}".format(np.mean(list_test_acc), np.std(list_test_acc)))
    list_test_ari = np.array(list_test_ari)
    print("Average test ARI: {:.3f} +/- {:.3f}".format(np.mean(list_test_ari), np.std(list_test_ari)))
    list_test_nmi = np.array(list_test_nmi)
    print("Average test NMI: {:.3f} +/- {:.3f}".format(np.mean(list_test_nmi), np.std(list_test_nmi)))
else:
    list_acc = np.array(list_acc)
    print("Average ACC: {:.3f} +/- {:.3f}".format(np.mean(list_acc), np.std(list_acc)))
    list_ari = np.array(list_ari)
    print("Average ARI: {:.3f} +/- {:.3f}".format(np.mean(list_ari), np.std(list_ari)))
    list_nmi = np.array(list_nmi)
    print("Average NMI: {:.3f} +/- {:.3f}".format(np.mean(list_nmi), np.std(list_nmi)))
