#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 10:11:05 2018

@author: maziarmoradifard
"""

#!/usr/bin/env python3

__author__ = "Thibaut Thonet, Maziar Moradi Fard"
__license__ = "GPL"

import tensorflow as tf
from utils import TF_FLOAT_TYPE

def fc_layers(input, specs):
    [dimensions, activations, names] = specs
    for dimension, activation, name in zip(dimensions, activations, names):
        input = tf.layers.dense(inputs=input, units=dimension, activation=activation, name=name, bias_initializer=tf.glorot_uniform_initializer(), reuse=tf.AUTO_REUSE)
    return input

def autoencoder(input, specs):
    [dimensions, activations, names] = specs
    mid_ind = int(len(dimensions)/2)

    # Encoder
    embedding = fc_layers(input, [dimensions[:mid_ind], activations[:mid_ind], names[:mid_ind]])
    # Decoder
    output = fc_layers(embedding, [dimensions[mid_ind:], activations[mid_ind:], names[mid_ind:]])

    return embedding, output

def f_func(x, y):
    #return tf.reduce_sum(tf.square(x - y), axis=1)
    return 1 - tf.reduce_sum(tf.multiply(x, y), axis=1) / (tf.norm(x, axis=1)*tf.norm(y, axis=1)) # Cosine dist.


def g_func(x, y):
    return tf.reduce_sum(tf.square(x - y), axis=1)
    #return 1 - tf.reduce_sum(tf.multiply(x, y), axis=1) / (tf.norm(x, axis=1)*tf.norm(y, axis=1)) # Cosine dist.


class CDKmCompGraph(object):
    """Computation graph for Deep K-Means
    """
    def __init__(self, ae_specs, keywords, n_clusters, val_lambda0, val_lambda1, cdc_version):
        input_size = ae_specs[0][-1]
        embedding_size = ae_specs[0][int((len(ae_specs[0])-1)/2)]
                    
        
            
        if cdc_version:
            
           # Placeholder tensor for input data
           self.input = tf.placeholder(dtype=TF_FLOAT_TYPE, shape=(None, input_size))

           # Auto-encoder loss computations
           self.embedding, self.output = autoencoder(self.input, ae_specs)  # Get the auto-encoder's embedding and output
           rec_error = g_func(self.input, self.output)  # Reconstruction error based on distance g
           
           #Cdc loss computation
           self.keywords = tf.constant(keywords, dtype=TF_FLOAT_TYPE)
           self.keywords_embedding, self.keywords_output = autoencoder(self.keywords, ae_specs)

           # k-Means loss computations
             ## Tensor for cluster representatives
           minval_rep, maxval_rep = -1, 1
           self.cluster_rep = tf.Variable(tf.random_uniform([n_clusters, embedding_size],
                                                    minval=minval_rep, maxval=maxval_rep,
                                                    dtype=TF_FLOAT_TYPE), name='cluster_rep', dtype=TF_FLOAT_TYPE)

           ## First, compute the distance f between the embedding and each cluster representative
           list_dist = []
           for i in range(0, n_clusters):
               dist = f_func(self.embedding, tf.reshape(self.cluster_rep[i, :], (1, embedding_size)))
               list_dist.append(dist)
           self.stack_dist = tf.stack(list_dist)

           ## Second, find the minimum squared distance for softmax normalization
           min_dist = tf.reduce_min(list_dist, axis=0)

           ## Third, compute exponentials with minimum argument of -100 to avoid underflow (0/0) issues in softmaxes
           self.alpha = tf.placeholder(dtype=TF_FLOAT_TYPE, shape=())  # Placeholder tensor for alpha
           list_exp = []
           for i in range(n_clusters):
               # exp = tf.exp(-tf.minimum(alpha * stack_dist[i], 80.0 * tf.ones(tf.shape(stack_dist[i]), TF_FLOAT_TYPE)))
               exp = tf.exp(-self.alpha * (self.stack_dist[i] - min_dist))
               list_exp.append(exp)
           # sum_exponentials += exp
           stack_exp = tf.stack(list_exp)
           sum_exponentials = tf.reduce_sum(stack_exp, axis=0)

           ## Fourth, compute softmaxes and the embedding/representative distances weighted by softmax
           list_softmax = []
           list_weighted_dist = []
           for j in range(n_clusters):
               softmax = stack_exp[j] / sum_exponentials
               weighted_dist = self.stack_dist[j] * softmax
               list_softmax.append(softmax)
               list_weighted_dist.append(weighted_dist)
           stack_weighted_dist = tf.stack(list_weighted_dist)
           
           

           # Compute the full loss combining the reconstruction error and k-means term
           self.ae_loss = tf.reduce_mean(rec_error)
           self.clustering_loss = tf.reduce_mean(tf.reduce_sum(stack_weighted_dist, axis=0))
           self.constrained_loss = tf.reduce_mean(f_func(self.keywords_embedding, self.cluster_rep))
           self.loss = self.ae_loss + val_lambda0 * self.clustering_loss + val_lambda1 * self.constrained_loss

           # The optimizer is defined to minimize this loss
           optimizer = tf.train.AdamOptimizer()
           self.pretrain_op = optimizer.minimize(self.ae_loss) # Pretrain the autoencoder before starting DKM
           self.train_op = optimizer.minimize(self.loss) # Train the whole DKM model
              
            
        else:
            
            # Placeholder tensor for input data
            self.input = tf.placeholder(dtype=TF_FLOAT_TYPE, shape=(None, input_size))

            # Auto-encoder loss computations
            self.embedding, self.output = autoencoder(self.input, ae_specs)  # Get the auto-encoder's embedding and output
            rec_error = g_func(self.input, self.output)  # Reconstruction error based on distance g
            self.keywords = tf.constant(keywords, dtype=TF_FLOAT_TYPE)
            self.keywords_embedding, _ = autoencoder(self.keywords, ae_specs)
            self.data_masked = tf.placeholder(dtype=TF_FLOAT_TYPE, shape=(None, input_size))
            self.data_second = tf.placeholder(dtype=TF_FLOAT_TYPE, shape=(None, input_size))
            # k-Means loss computations
            ## Tensor for cluster representatives
            minval_rep, maxval_rep = -1, 1
            self.cluster_rep = tf.Variable(tf.random_uniform([n_clusters, embedding_size],
                                                    minval=minval_rep, maxval=maxval_rep,
                                                    dtype=TF_FLOAT_TYPE), name='cluster_rep', dtype=TF_FLOAT_TYPE)

            ## First, compute the distance f between the embedding and each cluster representative
            list_dist = []
            for i in range(0, n_clusters):
                dist = f_func(self.embedding, tf.reshape(self.cluster_rep[i, :], (1, embedding_size)))
                list_dist.append(dist)
            self.stack_dist = tf.stack(list_dist)

            ## Second, find the minimum squared distance for softmax normalization
            min_dist = tf.reduce_min(list_dist, axis=0)

            ## Third, compute exponentials with minimum argument of -100 to avoid underflow (0/0) issues in softmaxes
            self.alpha = tf.placeholder(dtype=TF_FLOAT_TYPE, shape=())  # Placeholder tensor for alpha
            list_exp = []
            for i in range(n_clusters):
                # exp = tf.exp(-tf.minimum(alpha * stack_dist[i], 80.0 * tf.ones(tf.shape(stack_dist[i]), TF_FLOAT_TYPE)))
                exp = tf.exp(-self.alpha * (self.stack_dist[i] - min_dist))
                list_exp.append(exp)
                # sum_exponentials += exp
            stack_exp = tf.stack(list_exp)
            sum_exponentials = tf.reduce_sum(stack_exp, axis=0)

            ## Fourth, compute softmaxes and the embedding/representative distances weighted by softmax
            list_softmax = []
            list_weighted_dist = []
            for j in range(n_clusters):
                softmax = stack_exp[j] / sum_exponentials
                weighted_dist = self.stack_dist[j] * softmax
                list_softmax.append(softmax)
                list_weighted_dist.append(weighted_dist)
            stack_weighted_dist = tf.stack(list_weighted_dist)
            
            #Now the regularization part
            self.data_masked_embedding, _ = autoencoder(self.data_masked, ae_specs)
            self.data_second_embedding, _ = autoencoder(self.data_second, ae_specs)
            regularization_error = f_func(self.data_second_embedding, self.data_masked_embedding)

            

            # Compute the full loss combining the reconstruction error and k-means term
            self.clustering_loss = tf.reduce_mean(tf.reduce_sum(stack_weighted_dist, axis=0))
            self.constrained_loss = tf.reduce_mean(regularization_error)
            # Lexical pretraining
            self.ae_loss = tf.reduce_mean(rec_error) + val_lambda1 * self.constrained_loss
            self.loss = tf.reduce_mean(rec_error) + val_lambda0 * self.clustering_loss + val_lambda1 * self.constrained_loss
            
            # The optimizer is defined to minimize this loss
            optimizer = tf.train.AdamOptimizer()
            self.pretrain_op = optimizer.minimize(self.ae_loss) # Pretrain the autoencoder before starting DKM
            self.train_op = optimizer.minimize(self.loss) # Train the whole DKM model 
            
class CDKMCombined(object):
    """Computation graph for Deep K-Means
    
    """
    def __init__(self, ae_specs, keywords, n_clusters, val_lambda0, val_lambda1, val_lambda2, val_lambda3, val_lambda):
            input_size = ae_specs[0][-1]
            embedding_size = ae_specs[0][int((len(ae_specs[0])-1)/2)]
            self.input = tf.placeholder(dtype=TF_FLOAT_TYPE, shape=(None, input_size))

            # Auto-encoder loss computations
            self.embedding, self.output = autoencoder(self.input, ae_specs)  # Get the auto-encoder's embedding and output
            rec_error = g_func(self.input, self.output)  # Reconstruction error based on distance g
            self.keywords = tf.constant(keywords, dtype=TF_FLOAT_TYPE)
            self.keywords_embedding, _ = autoencoder(self.keywords, ae_specs)
            self.data_masked = tf.placeholder(dtype=TF_FLOAT_TYPE, shape=(None, input_size))
            self.data_second = tf.placeholder(dtype=TF_FLOAT_TYPE, shape=(None, input_size))
            # k-Means loss computations
            ## Tensor for cluster representatives
            minval_rep, maxval_rep = -1, 1
            self.cluster_rep = tf.Variable(tf.random_uniform([n_clusters, embedding_size],
                                                    minval=minval_rep, maxval=maxval_rep,
                                                    dtype=TF_FLOAT_TYPE), name='cluster_rep', dtype=TF_FLOAT_TYPE)

            ## First, compute the distance f between the embedding and each cluster representative
            list_dist = []
            for i in range(0, n_clusters):
                dist = f_func(self.embedding, tf.reshape(self.cluster_rep[i, :], (1, embedding_size)))
                list_dist.append(dist)
            self.stack_dist = tf.stack(list_dist)

            ## Second, find the minimum squared distance for softmax normalization
            min_dist = tf.reduce_min(list_dist, axis=0)

            ## Third, compute exponentials with minimum argument of -100 to avoid underflow (0/0) issues in softmaxes
            self.alpha = tf.placeholder(dtype=TF_FLOAT_TYPE, shape=())  # Placeholder tensor for alpha
            list_exp = []
            for i in range(n_clusters):
                # exp = tf.exp(-tf.minimum(alpha * stack_dist[i], 80.0 * tf.ones(tf.shape(stack_dist[i]), TF_FLOAT_TYPE)))
                exp = tf.exp(-self.alpha * (self.stack_dist[i] - min_dist))
                list_exp.append(exp)
                # sum_exponentials += exp
            stack_exp = tf.stack(list_exp)
            sum_exponentials = tf.reduce_sum(stack_exp, axis=0)

            ## Fourth, compute softmaxes and the embedding/representative distances weighted by softmax
            list_softmax = []
            list_weighted_dist = []
            for j in range(n_clusters):
                softmax = stack_exp[j] / sum_exponentials
                weighted_dist = self.stack_dist[j] * softmax
                list_softmax.append(softmax)
                list_weighted_dist.append(weighted_dist)
            stack_weighted_dist = tf.stack(list_weighted_dist)
            
            #Now the regularization part
            self.data_masked_embedding, _ = autoencoder(self.data_masked, ae_specs)
            self.data_second_embedding, _ = autoencoder(self.data_second, ae_specs)
            regularization_error = f_func(self.data_second_embedding, self.data_masked_embedding)

            

            # Compute the full loss combining the reconstruction error and k-means term
            self.clustering_loss = tf.reduce_mean(tf.reduce_sum(stack_weighted_dist, axis=0))
            self.constrained_loss_cdkm = tf.reduce_mean(regularization_error)
            self.constrained_loss_cdc = tf.reduce_mean(f_func(self.keywords_embedding, self.cluster_rep))
            # Lexical pretraining
            self.ae_loss = tf.reduce_mean(rec_error)
            self.loss_cdkm = self.ae_loss + val_lambda0 * self.clustering_loss + val_lambda1 * self.constrained_loss_cdkm 
            self.loss_cdc = self.ae_loss + val_lambda2 * self.clustering_loss + val_lambda3 * self.constrained_loss_cdc
            self.loss = val_lambda * self.loss_cdc + (1-val_lambda) * self.loss_cdkm
            
            # The optimizer is defined to minimize this loss
            optimizer = tf.train.AdamOptimizer()
            self.pretrain_op = optimizer.minimize(self.ae_loss) # Pretrain the autoencoder before starting DKM
            self.train_op = optimizer.minimize(self.loss) # Train the whole DKM model                         
class DkmCompGraph(object):
    """Computation graph for Deep K-Means
    """

    def __init__(self, ae_specs, n_clusters, val_lambda):
        input_size = ae_specs[0][-1]
        embedding_size = ae_specs[0][int((len(ae_specs[0])-1)/2)]

        # Placeholder tensor for input data
        self.input = tf.placeholder(dtype=TF_FLOAT_TYPE, shape=(None, input_size))

        # Auto-encoder loss computations
        self.embedding, self.output = autoencoder(self.input, ae_specs)  # Get the auto-encoder's embedding and output
        rec_error = g_func(self.input, self.output)  # Reconstruction error based on distance g

        # k-Means loss computations
        ## Tensor for cluster representatives
        minval_rep, maxval_rep = -1, 1
        self.cluster_rep = tf.Variable(tf.random_uniform([n_clusters, embedding_size],
                                                    minval=minval_rep, maxval=maxval_rep,
                                                    dtype=TF_FLOAT_TYPE), name='cluster_rep', dtype=TF_FLOAT_TYPE)

        ## First, compute the distance f between the embedding and each cluster representative
        list_dist = []
        for i in range(0, n_clusters):
            dist = f_func(self.embedding, tf.reshape(self.cluster_rep[i, :], (1, embedding_size)))
            list_dist.append(dist)
        self.stack_dist = tf.stack(list_dist)

        ## Second, find the minimum squared distance for softmax normalization
        min_dist = tf.reduce_min(list_dist, axis=0)

        ## Third, compute exponentials with minimum argument of -100 to avoid underflow (0/0) issues in softmaxes
        self.alpha = tf.placeholder(dtype=TF_FLOAT_TYPE, shape=())  # Placeholder tensor for alpha
        list_exp = []
        for i in range(n_clusters):
            # exp = tf.exp(-tf.minimum(alpha * stack_dist[i], 80.0 * tf.ones(tf.shape(stack_dist[i]), TF_FLOAT_TYPE)))
            exp = tf.exp(-self.alpha * (self.stack_dist[i] - min_dist))
            list_exp.append(exp)
            # sum_exponentials += exp
        stack_exp = tf.stack(list_exp)
        sum_exponentials = tf.reduce_sum(stack_exp, axis=0)

        ## Fourth, compute softmaxes and the embedding/representative distances weighted by softmax
        list_softmax = []
        list_weighted_dist = []
        for j in range(n_clusters):
            softmax = stack_exp[j] / sum_exponentials
            # softmax = tf.div(stack_exp[j], sum_exponentials)
            weighted_dist = self.stack_dist[j] * softmax
            # weighted_dist = tf.multiply(stack_dist[j], softmax)
            list_softmax.append(softmax)
            list_weighted_dist.append(weighted_dist)
        stack_weighted_dist = tf.stack(list_weighted_dist)

        # Compute the full loss combining the reconstruction error and k-means term
        self.ae_loss = tf.reduce_mean(rec_error)
        self.kmeans_loss = tf.reduce_mean(tf.reduce_sum(stack_weighted_dist, axis=0))
        self.loss = self.ae_loss + val_lambda * self.kmeans_loss

        # The optimizer is defined to minimize this loss
        optimizer = tf.train.AdamOptimizer()
        self.pretrain_op = optimizer.minimize(self.ae_loss) # Pretrain the autoencoder before starting DKM
        self.train_op = optimizer.minimize(self.loss) # Train the whole DKM model

class AeCompGraph(object):
    """Computation graph for a fully-connected auto-encoder
    """

    def __init__(self, ae_specs):
        input_size = ae_specs[0][-1]

        # Placeholder tensor for input data
        self.input = tf.placeholder(dtype=TF_FLOAT_TYPE, shape=(None, input_size))

        # Auto-encoder loss computations
        self.embedding, self.output = autoencoder(self.input, ae_specs)  # Get the auto-encoder's embedding and output
        rec_error = g_func(self.input, self.output)  # Reconstruction error based on distance g

        # Compute the full loss combining the reconstruction error and k-means term
        self.loss = tf.reduce_mean(rec_error)

        # The optimizer is defined to minimize this loss
        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(self.loss) # Train the auto-encoder

class DcnCompGraph(object):
    """Computation graph for the Deep Clustering Network model proposed in:
        Yang, B., Fu, X., Sidiropoulos, N. D., & Hong, M. (2017). Towards K-means-friendly Spaces: Simultaneous Deep
        Learning and Clustering. In ICML '17 (pp. 3861–3870).
    This implementation is inspired by https://github.com/boyangumn/DCN
    """

    def __init__(self, ae_specs, n_clusters, batch_size, n_samples, val_lambda):
        input_size = ae_specs[0][-1]
        embedding_size = ae_specs[0][int((len(ae_specs[0]) - 1) / 2)]

        # Placeholder tensor for input data
        self.input = tf.placeholder(dtype=TF_FLOAT_TYPE, shape=(batch_size, input_size))

        # Auto-encoder loss computations
        self.embedding, self.output = autoencoder(self.input, ae_specs)  # Get the auto-encoder's embedding and output
        rec_error = g_func(self.input, self.output)  # Reconstruction error based on distance g
        self.ae_loss = tf.reduce_mean(rec_error)

        # Clustering loss computations
        ## Tensor for cluster representatives
        minval_rep, maxval_rep = -1, 1
        self.cluster_rep = tf.Variable(tf.random_uniform([n_clusters, embedding_size],
                                                    minval=minval_rep, maxval=maxval_rep,
                                                    dtype=TF_FLOAT_TYPE), name='cluster_rep', dtype=TF_FLOAT_TYPE)

        ## Clustering assignments for all samples in the dataset
        initial_clustering_assign = tf.random_uniform(minval=0, maxval=n_clusters, dtype=tf.int32, shape=[n_samples])
        self.cluster_assign = tf.Variable(initial_clustering_assign, name='cluster_assign', dtype=tf.int32, trainable=False)

        ## Get the cluster representative corresponding to the cluster of each batch sample
        self.indices = tf.placeholder(dtype=tf.int32, shape=batch_size)  # Placeholder for sample indices in current batch
        batch_clust_rep = []
        for j in range(batch_size):
            k = self.cluster_assign[self.indices[j]]  # Clustering assignment for sample j in batch
            batch_clust_rep.append(self.cluster_rep[k, :])
        stack_batch_clust_rep = tf.stack(batch_clust_rep)

        ## Compute the k-means term
        clustering_error = f_func(self.embedding, stack_batch_clust_rep)

        # Compute the full loss combining the reconstruction error and k-means term
        self.ae_loss = tf.reduce_mean(rec_error)
        self.kmeans_loss = tf.reduce_mean(clustering_error)
        self.loss = self.ae_loss + val_lambda * self.kmeans_loss

        # The optimizer is defined to minimize this loss
        optimizer = tf.train.AdamOptimizer()
        self.pretrain_op = optimizer.minimize(self.ae_loss) # Pretrain the autoencoder before starting DCN
        self.train_op = optimizer.minimize(self.loss)

        # As it has been pointed out in the DCN paper, first the weights of the autoencoder shall be trained then the
        # assignments and cluster representatives. So far in the computation graph we have optimized the weights of the
        # autoencoder, now it's the time for the assignments and representatives.

        # Update the clustering assignments
        for j in range(batch_size):
            # Find which cluster representative is the closest to the current batch sample
            new_assign = tf.argmin(f_func(tf.reshape(self.embedding[j, :], (1, embedding_size)), self.cluster_rep),
                                   output_type=tf.int32)
            # Update the clustering assignment
            self.cluster_assign_update = tf.assign(self.cluster_assign[self.indices[j]], new_assign)

        # Update the cluster representatives
        ## Initialize the value of count
        initial_count = tf.constant(100.0, shape=[n_clusters])
        count = tf.Variable(initial_count, name='count', dtype=TF_FLOAT_TYPE, trainable=False)
        ## Update the cluster representatives according to Equation (8) in the DCN paper
        for j in range(batch_size):
            k = self.cluster_assign[self.indices[j]]  # Clustering assignment for sample j in batch
            self.count_update = tf.assign(count[k], count[k] + 1)  # Updated count for cluster assignments
            new_rep = self.cluster_rep[k] - (1 / count[k]) * (self.cluster_rep[k] - self.embedding[j])
            self.cluster_rep_update = tf.assign(self.cluster_rep[k], new_rep)