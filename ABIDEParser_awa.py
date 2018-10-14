# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import os

import numpy as np
from scipy import sparse as sp
import random
import pandas as pd

# Reading and computing the input data

# Selected pipeline
pipeline = 'cpac'

# Input data variables
#root_folder = '/home/arvind/deeplearning/population-gcn/'
#data_folder = os.path.join(root_folder, 'ABIDE_pcp/cpac/filt_noglobal')
#phenotype = os.path.join(root_folder, 'ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')

num_nodes = 564
num_classes = 3
num_features = 354
no_of_aff = 3 #!--
ground_truth = []


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def get_features():
    df = pd.read_excel('tadpole-preprocessed.xlsx')
    df = np.array(df)
    row,col = np.shape(df)
    ##print row
    ##print col
    ground_truth = df[:,10].astype(int)
    ## Balancing the dataset

    ##print ground_truth
    #print "max and min of ground truth", max(ground_truth), min(ground_truth)
    for i in range(col):
        if df[1, i] == "1.5 Tesla MRI" or df[1, i] == "3 Tesla MRI":
            for r in range(row):
                if df[r, i] == "1.5 Tesla MRI":
                    df[r, i] = 1.5
                elif df[r, i] == "3 Tesla MRI":
                    df[r, i] = 3
        if df[1, i] == "Pass" or df[1, i] == "Fail":
            for r in range(row):
                if df[r, i] == "Pass":
                    df[r, i] = 1
                elif df[r, i] == "Fail":
                    df[r, i] = 0

    #features = df[:,19:]
    features = df[:, 18:]
    _,fcol = np.shape(features)
    for j in range(0,  fcol):
        maxi = max(features[:, j])
        mini = min(features[:, j])
        if maxi != mini:
            for i in range(row):
                features -= mini
                features[i, j] /= (maxi - mini)
        else:
            for i in range(row):
                features[i, j] = 1
    c1_idx = [i for i in range(num_nodes) if ground_truth[i] == 1]
    c2_idx = [i for i in range(num_nodes) if ground_truth[i] == 2]
    c3_idx = [i for i in range(num_nodes) if ground_truth[i] == 3]
    np.random.shuffle(c1_idx)
    np.random.shuffle(c2_idx)
    np.random.shuffle(c3_idx)
    c1_idx = c1_idx[:100]
    c2_idx = c2_idx[:120]
    c3_idx = c3_idx[:80]
    all_idx = np.concatenate([c1_idx, c2_idx, c3_idx], axis=0)
    features = features[all_idx, :]
    ground_truth = ground_truth[all_idx]
    #print np.shape(features),"shape of features"
    #exit()
    return ground_truth, features, all_idx


def get_affinity(sparse_graph, idx):
    df = pd.read_excel('tadpole-preprocessed.xlsx')
    df = np.array(df)
    num_nodes, col = np.shape(df)
    age = df[:,11]
    gender = df[:,12]
    # fdg = df[:, 18]
    apoe = df[:, 17]
    graph = np.zeros((no_of_aff, num_nodes, num_nodes))
    #print "max,min of age: ", max(age), min(age)
    for i in range(num_nodes):
         for j in range(i+1, num_nodes):
             if gender[i] == gender[j]:
                graph[2, i, j] += 1
                graph[2, j, i] += 1
    np.savetxt("gender.csv", graph[1], delimiter=',')

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.absolute(age[i] - age[j]) < 2:
                graph[1, i, j] += 1
                graph[1, j, i] += 1
    np.savetxt("age.csv", graph[2], delimiter=',')

    # for i in range(num_nodes):
    #     for j in range(i + 1, num_nodes):
    #         if np.absolute(fdg[i] - fdg[j]) < 0.10:
    #             graph[2, i, j] += 1
    #             graph[2, j, i] += 1
    # np.savetxt("fdg.csv", graph[2], delimiter=',')

    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if apoe[i] == apoe[j]:
                graph[0, i, j] += 1
                graph[0, j, i] += 1

    np.savetxt("apoe.csv", graph[0], delimiter=',')

    new_num_nodes = idx.shape[0]
    new_graph = np.zeros((no_of_aff, new_num_nodes, new_num_nodes))
    for i in range(no_of_aff):
        tmp_graph = graph[i, idx, :]
        tmp_graph = tmp_graph[:, idx]
        new_graph[i] = tmp_graph * sparse_graph

    return new_graph
