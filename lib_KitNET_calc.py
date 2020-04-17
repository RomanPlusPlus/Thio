""" Almost everything related to KitNET is done in this module.

KitNET is an online anomaly detection algorithm based on an ensemble of autoencoders, invented by [Mirsky et al, 2017].

Most of the code below is a modified version of their code, released under the MIT license.
Source: https://github.com/ymirsky/KitNET-py
Paper: Yisroel Mirsky, Tomer Doitshman, Yuval Elovici, and Asaf Shabtai, "Kitsune: An Ensemble of Autoencoders for
Online Network Intrusion Detection", Network and Distributed System Security Symposium 2018 (NDSS'18)
https://arxiv.org/abs/1802.09089
A small part of the code is a modified version of the code by [Yusuke, 2017], released under the MIT license.
Source: https://github.com/yusugomori/DeepLearning .
The corresponding license texts are at end of this file.
"""

import numpy as np
import time
from scipy.cluster.hierarchy import linkage, to_tree
import pickle

from helper_funcs import append_logs, get_full_path, synthetic_data7

use_synthetic_data7 = synthetic_data7()

name4logs = "lib_KitNET_calc"

msg = "Reading Sample dataset..."
append_logs(msg, name4logs, "verbose")

if use_synthetic_data7:
    filename = "dataset/syntheticData.txt"
else:
    filename = "dataset/fetchedData.txt"

np.seterr(all='ignore')


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


class DenoisingAutoencoderParams:
    """A data class for storing the Denoising Autoencoder params."""

    def __init__(self, n_visible=5, n_hidden=3, lr=0.001, corruption_level=0.0, grace_period=10000, hidden_ratio=None):
        self.n_visible = n_visible  # num of units in visible (input) layer
        self.n_hidden = n_hidden  # num of units in hidden layer
        self.lr = lr
        self.corruption_level = corruption_level
        self.gracePeriod = grace_period
        self.hiddenRatio = hidden_ratio


class DenoisingAutoencoder:
    """This class represents Denoising Autoencoder.

    Autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner.
    For a popular introduction into Autoencoders, see: https://en.wikipedia.org/wiki/Autoencoder

    For Denoising Autoencoders, see:
    Vincent et al, 2008. Extracting and Composing Robust Features with Denoising Autoencoders
    Yusuke Sugomori, 2013. Stochastic Gradient Descent for Denoising Autoencoders,http://yusugomori.com/docs/SGD_DA.pdf


    """

    def __init__(self, params):
        self.params = params

        if self.params.hiddenRatio is not None:
            self.params.n_hidden = int(np.ceil(self.params.n_visible * self.params.hiddenRatio))

        # for 0-1 normlaization
        self.norm_max = np.ones((self.params.n_visible,)) * -np.Inf
        self.norm_min = np.ones((self.params.n_visible,)) * np.Inf
        self.n = 0

        self.rng = np.random.RandomState(1234)

        a = 1. / self.params.n_visible
        self.W = np.array(self.rng.uniform(  # initialize W uniformly
            low=-a,
            high=a,
            size=(self.params.n_visible, self.params.n_hidden)))

        self.hbias = np.zeros(self.params.n_hidden)  # initialize h bias 0
        self.vbias = np.zeros(self.params.n_visible)  # initialize v bias 0
        self.W_prime = self.W.T

    def get_corrupted_input(self, g_input, corruption_level):
        assert corruption_level < 1

        return self.rng.binomial(size=g_input.shape,
                                 n=1,
                                 p=1 - corruption_level) * g_input

    # Encode
    def get_hidden_values(self, e_input):
        return sigmoid(np.dot(e_input, self.W) + self.hbias)

    # Decode
    def get_reconstructed_input(self, hidden):
        return sigmoid(np.dot(hidden, self.W_prime) + self.vbias)

    def train(self, x):
        self.n = self.n + 1
        # update norms
        self.norm_max[x > self.norm_max] = x[x > self.norm_max]
        self.norm_min[x < self.norm_min] = x[x < self.norm_min]

        # 0-1 normalize
        x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)

        if self.params.corruption_level > 0.0:
            tilde_x = self.get_corrupted_input(x, self.params.corruption_level)
        else:
            tilde_x = x
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        l_h2 = x - z
        l_h1 = np.dot(l_h2, self.W) * y * (1 - y)

        l_vbias = l_h2
        l_hbias = l_h1
        l_w = np.outer(tilde_x.T, l_h1) + np.outer(l_h2.T, y)

        self.W += self.params.lr * l_w
        self.hbias += self.params.lr * np.mean(l_hbias, axis=0)
        self.vbias += self.params.lr * np.mean(l_vbias, axis=0)
        return np.sqrt(np.mean(l_h2 ** 2))  # the RMSE reconstruction error during training

    def reconstruct(self, x):
        y = self.get_hidden_values(x)
        z = self.get_reconstructed_input(y)
        return z

    def execute(self, x):  # returns MSE of the reconstruction of x
        if self.n < self.params.gracePeriod:
            return 0.0
        else:
            # 0-1 normalize
            x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)
            z = self.reconstruct(x)
            rmse = np.sqrt(((x - z) ** 2).mean())  # MSE
            return rmse

    def inGrace(self):
        return self.n < self.params.gracePeriod


class CorClust:
    """ A helper class for KitNET which performs a correlation-based incremental clustering of the dimensions in X
    n: the number of dimensions in the dataset

    For more information and citation, please see the NDSS'18 paper:
    Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection
    """

    def __init__(self, n):
        # parameter:
        self.n = n
        # varaibles
        self.c = np.zeros(n)  # linear num of features
        self.c_r = np.zeros(n)  # linear sum of feature residules
        self.c_rs = np.zeros(n)  # linear sum of feature residules
        self.C = np.zeros((n, n))  # partial correlation matrix
        self.N = 0  # number of updates performed

    # x: a numpy vector of length n
    def update(self, x):
        self.N += 1
        self.c += x
        c_rt = x - self.c / self.N
        self.c_r += c_rt
        self.c_rs += c_rt ** 2
        self.C += np.outer(c_rt, c_rt)

    # creates the current correlation distance matrix between the features
    def corrDist(self):
        c_rs_sqrt = np.sqrt(self.c_rs)
        big_c_rs_sqrt = np.outer(c_rs_sqrt, c_rs_sqrt)
        big_c_rs_sqrt[
            big_c_rs_sqrt == 0] = 1e-100  # protects against dive by zero erros(occurs when a feature is a constant)
        dist = 1 - self.C / big_c_rs_sqrt  # the correlation distance matrix
        dist[
            # small negatives may appear due to the incremental fashion in which we update the mean.
            # Therefore, we 'fix' them
            dist < 0] = 0
        return dist

    # clusters the features together, having no more than maxClust features per cluster
    def cluster(self, max_clust):
        corr_dist = self.corrDist()
        linkage_matrix = linkage(corr_dist[np.triu_indices(self.n, 1)])  # a linkage matrix based on the distance matrix
        if max_clust < 1:
            max_clust = 1
        if max_clust > self.n:
            max_clust = self.n
        cluster_map = self.__breakClust__(to_tree(linkage_matrix), max_clust)
        return cluster_map

    # a recursive helper function which breaks down the dendrogram branches until
    # all clusters have no more than maxClust elements
    def __breakClust__(self, dendro, max_clust):
        if dendro.count <= max_clust:  # base case: we found a minimal cluster, so mark it
            return [dendro.pre_order()]  # return the origional ids of the features in this cluster
        return self.__breakClust__(dendro.get_left(), max_clust) + self.__breakClust__(dendro.get_right(), max_clust)


class KitNET:
    """This class represents a KitNET machine learner.

    # n: the number of features in your input dataset (i.e., x \in R^n)
    # m: the maximum size of any autoencoder in the ensemble layer
    # AD_grace_period: the number of instances the network will learn from before producing anomaly scores
    # FM_grace_period: the number of instances which will be taken to learn the feature mapping. If 'None',
    # then FM_grace_period=AM_grace_period.
    # learning_rate: the default stochastic gradient descent learning rate for all autoencoders in the KitNET instance.
    # hidden_ratio: the default ratio of hidden to visible neurons. E.g., 0.75 will cause roughly a 25% compression in
    # the hidden layer.
    # feature_map: One may optionally provide a feature map instead of learning one. The map must be a list, where the
    # i-th entry contains a list of the feature indices to be assingned to the i-th autoencoder in the ensemble.
    # For example, [[2,5,3],[4,0,1],[6,7]]

    For more information and citation, please see the NDSS'18 paper:
    Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection
    """

    def __init__(self, n, max_autoencoder_size=10, fm_grace_period=None, ad_grace_period=10000, learning_rate=0.1,
                 hidden_ratio=0.75, feature_map=None):
        # Parameters:
        self.AD_grace_period = ad_grace_period
        if fm_grace_period is None:
            self.FM_grace_period = ad_grace_period
        else:
            self.FM_grace_period = fm_grace_period
        if max_autoencoder_size <= 0:
            self.m = 1
        else:
            self.m = max_autoencoder_size
        self.lr = learning_rate
        self.hr = hidden_ratio
        self.n = n

        # Variables
        self.n_trained = 0  # the number of training instances so far
        self.n_executed = 0  # the number of executed instances so far
        self.v = feature_map
        if self.v is None:
            append_logs("Feature-Mapper: train-mode, Anomaly-Detector: off-mode", "KitNET lib", "verbose")
        else:
            self.__createAD__()
            append_logs("Feature-Mapper: execute-mode, Anomaly-Detector: train-mode", "KitNET lib", "verbose")
        self.FM = CorClust(self.n)  # incremental feature cluatering for the feature mapping process
        self.ensembleLayer = []
        self.outputLayer = None

    # If FM_grace_period+AM_grace_period has passed, then this function executes KitNET on x.
    # Otherwise, this function learns from x.
    # x: a numpy array of length n
    # Note: KitNET automatically performs 0-1 normalization on all attributes.
    def process(self, x):
        if self.n_trained > self.FM_grace_period + self.AD_grace_period:  # If both the FM and AD are in execute-mode
            return self.execute(x)
        else:
            self.train(x)
            return 0.0

    # force train KitNET on x
    # returns the anomaly score of x during training (do not use for alerting)
    def train(self, x):
        # If the FM is in train-mode, and the user has not supplied a feature mapping
        if self.n_trained <= self.FM_grace_period and self.v is None:
            # update the incremetnal correlation matrix
            self.FM.update(x)
            if self.n_trained == self.FM_grace_period:  # If the feature mapping should be instantiated
                self.v = self.FM.cluster(self.m)
                self.__createAD__()
                t_msg = "The Feature-Mapper found a mapping: " + str(self.n) + " features to " + str(
                    len(self.v)) + " autoencoders."
                append_logs(t_msg, "KitNET lib", "verbose")
                t_msg = "Feature-Mapper: execute-mode, Anomaly-Detector: train-mode"
                append_logs(t_msg, "KitNET lib", "verbose")
        else:  # train
            # Ensemble Layer
            s_l1 = np.zeros(len(self.ensembleLayer))
            for a in range(len(self.ensembleLayer)):
                # make sub instance for autoencoder 'a'
                xi = x[self.v[a]]
                s_l1[a] = self.ensembleLayer[a].train(xi)
            # OutputLayer
            self.outputLayer.train(s_l1)
            if self.n_trained == self.AD_grace_period + self.FM_grace_period:
                t_msg = "Feature-Mapper: execute-mode, Anomaly-Detector: exeute-mode"
                append_logs(t_msg, "KitNET lib", "verbose")
        self.n_trained += 1
        return self

        # force execute KitNET on x

    def execute(self, x):
        if self.v is None:
            e_msg = 'cant execute x, because a feature mapping not learned or provided. Try running process(x) instead.'
            append_logs(e_msg, "KitNET lib", "always")
            raise RuntimeError(msg)
        else:
            self.n_executed += 1
            # Ensemble Layer
            s_l1 = np.zeros(len(self.ensembleLayer))
            for a in range(len(self.ensembleLayer)):
                # make sub inst
                xi = x[self.v[a]]
                s_l1[a] = self.ensembleLayer[a].execute(xi)
            # OutputLayer
            return self.outputLayer.execute(s_l1)

    def __createAD__(self):
        # construct ensemble layer
        for ad_map in self.v:
            params = DenoisingAutoencoderParams(n_visible=len(ad_map), n_hidden=0, lr=self.lr, corruption_level=0,
                                                grace_period=0, hidden_ratio=self.hr)
            self.ensembleLayer.append(DenoisingAutoencoder(params))

        # construct output layer
        params = DenoisingAutoencoderParams(len(self.v), n_hidden=0, lr=self.lr, corruption_level=0,
                                            grace_period=0, hidden_ratio=self.hr)
        self.outputLayer = DenoisingAutoencoder(params)

    def pickle(self):

        timestamp = time.time()
        obj_to_save = (self, timestamp)
        with open(get_full_path('pickled_models/test_file.pkl'), 'wb') as pkl:
            pickle.dump(obj_to_save, pkl)


def get_model(input_dataframe):
    input_arr = input_dataframe.to_numpy()

    dataset_size = len(input_dataframe.index)

    # KitNET params:
    max_ae = 10  # maximum size for any autoencoder in the ensemble layer

    fm_grace = int(
        dataset_size * 0.1)  # the number of instances taken to learn the feature mapping (the ensemble's architecture)
    ad_grace = dataset_size - fm_grace  # the number of instances used to train the anomaly detector (ensemble itself)

    append_logs("Dataset_size: " + str(dataset_size) + " . FMgrace: " + str(fm_grace) + " . ADgrace: " + str(ad_grace),
                name4logs, "verbose")

    append_logs("numpy.ndarray tail my input_arr:\n" + str(input_arr[-3:]), name4logs, "verbose")

    # Build KitNET    
    kit_net_obj = KitNET(input_arr.shape[1], max_ae, fm_grace, ad_grace)

    model = None
    for i in range(input_arr.shape[0]):
        if i % 1000 == 0:
            g_msg = "progress: " + str(i)
            # save_model_to_pickle(model, -1, "pickled_models/kitnet_test_" + str(i) + ".pkl")
            append_logs(g_msg, name4logs, "verbose")
        model = kit_net_obj.train(input_arr[i, ])

    return model, None, True


def ask_model(lmodel, observations_df, scaling):  # TODO: use scaling for KitNET too
    datapoint = None
    try:
        datapoint = observations_df.to_numpy()[-1]
        rmse_score = lmodel.execute(datapoint)
    except Exception as e:
        rmse_score = 0
        append_logs("ERROR: KitNET ask_model failed. datapoint: " + str(datapoint) + " . Exception: " + str(e),
                    name4logs, "always")
    return rmse_score

# ------------------------------------

# Most of the code above is a modified version of the code by 2017 Yisroel Mirsky, released under the MIT license.
# A small part of the code is a modified version of the code by 2017 Yusuke Sugomori, released under the MIT license.
# The corresponding license texts are below.

# Copyright (c) 2017 Yusuke Sugomori
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Portions of this code have been adapted from Yusuke Sugomori's code on GitHub:
# https://github.com/yusugomori/DeepLearning

# ----------------------------------------------------------------------

# Copyright (c) 2017 Yisroel Mirsky
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Portions of this code have been adapted from Yisroel Mirsky's code on GitHub:
# https://github.com/ymirsky/KitNET-py

# For more information and citation, see the NDSS'18 paper:
# Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection
