import os
import csv
import numpy as np
import scipy.io as sio

from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
from nilearn import connectome


# Load precomputed fMRI connectivity networks
def get_networks(subject_list, kind, atlas_name="aal", variable='connectivity'):
    """
        subject_list : list of subject IDs
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        atlas_name   : name of the parcellation atlas used
        variable     : variable name in the .mat file that has been used to save the precomputed networks


    return:
        matrix      : feature matrix of connectivity networks (num_subjects x network_size)
    """

    all_networks = []
    for subject in subject_list:
        fl = os.path.join(data_folder, subject,
                          subject + "_" + atlas_name + "_" + kind + ".mat")
        matrix = sio.loadmat(fl)[variable]
        all_networks.append(matrix)
    # all_networks=np.array(all_networks)

    idx = np.triu_indices_from(all_networks[0], 1)
    norm_networks = [np.arctanh(mat) for mat in all_networks]
    vec_networks = [mat[idx] for mat in norm_networks]
    matrix = np.vstack(vec_networks)

    return matrix

def site_percentage(train_ind, perc):
    """
        train_ind    : indices of the training samples
        perc         : percentage of training set used
        subject_list : list of subject IDs

    return:
        labeled_indices      : indices of the subset of training samples
    """
    with open('./labelShuffle.csv', 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        lst=[item[0].split(',') for item in spamreader]
    csv_labels = np.asarray([[float(ii) for ii in i] for i in lst])
    train_list = csv_labels[train_ind]
    # sites = np.array([0,1])
    # unique = np.unique(list(sites)).tolist()
    # site = np.array([unique(sites[train_list[x]]) for x in range(len(train_list))])


    # num_nodes = len(id_in_site)
    # labeled_num = int(round(perc * num_nodes))
    # labeled_indices.extend(train_ind[id_in_site[:labeled_num]])

    labeled_indices = []

    #for i in np.unique(site):
    #id_in_site = np.argwhere(site == i).flatten()
    num_nodes = 569
    labeled_num = int(round(perc * num_nodes))
    labeled_indices.extend(train_ind[:labeled_num])

    return labeled_indices


def get_labelChestXray():

    with open('./labelShuffle.csv', 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        lst=[item[0].split(',') for item in spamreader]
    csv_labels = np.asarray([[float(ii) for ii in i] for i in lst])
    num_nodes = len(lst)
    y_data = np.zeros([num_nodes, 2])
    # y_data = np.zeros([num_nodes, 1])
    y = np.zeros([num_nodes, 1])
    for i in range(num_nodes):
        y_data[i, int(csv_labels[i]) - 1] = 1
        # y_data[i] = int(labels[subject_IDs[i]]) - 1
        y[i] = int(csv_labels[i])
        #site[i] = unique.index(sites[subject_IDs[i]])
    return y_data, y, csv_labels
def get_genderGraphGen():

    with open('./genderShuffle.csv', 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        lst=[item[0].split(',') for item in spamreader]


    csv_Gen=np.asarray([[float(ii) for ii in i] for i in lst])
    num_nodes = len(lst)
    graph1 = np.zeros((num_nodes, num_nodes))

    w1 = 1.0
    for k in range(num_nodes):
        for j in range(k + 1, num_nodes):
            if csv_Gen[k] == csv_Gen[j]:
                graph1[k, j] += 1
                graph1[j, k] += 1
    w1 = 1.0
    w2 = 1.0
    graph = w1*graph1
    return graph

def get_genderGraphAge():
    with open('./ageShuffle.csv', 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        lst=[item[0].split(',') for item in spamreader]


    csv_age=np.asarray([[float(ii) for ii in i] for i in lst])
    num_nodes = len(lst)
    graph2 = np.zeros((num_nodes, num_nodes))

    #for l in csv_age:
        #label_dict = csv_labels[l]

        # # quantitative phenotypic scores
        # #if l in ['AGE_AT_SCAN', 'FIQ']:
    for k in range(num_nodes):
        for j in range(k + 1, num_nodes):
            try:
                val = abs(float(csv_age[k] ) - float(csv_age[j]))
                if val < 2:
                    graph2[k, j] += 1
                    graph2[j, k] += 1
            except ValueError:  # missing label
                pass

    w1 = 1.0
    w2 = 1.0
    graph =  w2*graph2
    return graph

def get_genderGraphAge2():
    with open('./ageShuffle.csv', 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        lst=[item[0].split(',') for item in spamreader]


    csv_age=np.asarray([[float(ii) for ii in i] for i in lst])
    num_nodes = len(lst)
    graph2 = np.zeros((num_nodes, num_nodes))

    #for l in csv_age:
        #label_dict = csv_labels[l]

        # # quantitative phenotypic scores
        # #if l in ['AGE_AT_SCAN', 'FIQ']:
    for k in range(num_nodes):
        for j in range(k + 1, num_nodes):
            try:
                val1 = abs(float(csv_age[k]))
                val2 = abs(float(csv_age[j]))
                if val1 <= 64 and val1> 25 and val2 <= 64 and val2> 25:
                    graph2[k, j] += 1
                    graph2[j, k] += 1
                elif val1 <= 25 and val2 <= 25:
                    graph2[k, j] += 1
                    graph2[j, k] += 1
                elif val1 >64 and val2 > 64:
                    graph2[k, j] += 1
                    graph2[j, k] += 1
            except ValueError:  # missing label
                pass
    print(graph2, "hello")
    #np.savetxt("agechestxray.csv", graph2, delimiter=',')

    w1 = 1.0
    w2 = 1.0
    graph =  w2*graph2
    return graph




def get_ChestXray():
    #with open('/home/leslie/Desktop/Features_ChestXray.csv', 'rb') as csvfile:
    with open('./featuresShuffle.csv','rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        lst=[item[0].split(',') for item in spamreader]

    csv_np=np.asarray([[float(ii) for ii in i] for i in lst])
    print('Read done...', csv_np.shape)
    return csv_np