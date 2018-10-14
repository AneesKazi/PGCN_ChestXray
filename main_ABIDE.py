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

import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import scipy.io as sio
import sklearn.metrics
import tensorflow as tf
from scipy import sparse
from scipy.spatial import distance
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

import ABIDEParser_awa as Reader
import train_GCN as Train

flags = tf.app.flags
FLAGS = flags.FLAGS


# Prepares the training/test data for each cross validation fold and trains the GCN
def train_fold(train_ind, test_ind, val_ind, graph_feat, features, y, y_data, params, subject_IDs, pathToSave, i, subject_labels, idx):
    """
        train_ind       : indices of the training samples
        test_ind        : indices of the test samples
        val_ind         : indices of the validation samples
        graph_feat      : population graph computed from phenotypic measures num_subjects x num_subjects
        features        : feature vectors num_subjects x num_features
        y               : ground truth labels (num_subjects x 1)
        y_data          : ground truth labels - different representation (num_subjects x 2)
        params          : dictionnary of GCNs parameters
        subject_IDs     : list of subject IDs

    returns:

        test_acc    : average accuracy over the test samples using GCNs
        test_auc    : average area under curve over the test samples using GCNs
        lin_acc     : average accuracy over the test samples using the linear classifier
        lin_auc     : average area under curve over the test samples using the linear classifier
        fold_size   : number of test samples
    """

    print(len(train_ind))
    tf.reset_default_graph()
    tf.app.flags._global_parser = argparse.ArgumentParser()

    # selection of a subset of data if running experiments with a subset of the training set
    # labeled_ind = Reader.site_percentage(train_ind, params['num_training'], subject_IDs)
    num_nodes = np.size(graph_feat, 0)
    #print features[0,:],"features"
    x_data_1 = features.astype(float)#Reader.feature_selection(features, y, labeled_ind, params['num_features'])
    xrow,xcol = np.shape(x_data_1)
    for i in range(xrow):
        for j in range(xcol):
            x_data_1[i, j] = round(x_data_1[i,j], 4)
    fold_size = len(test_ind)
    x_data_1[np.where(np.isnan(x_data_1))] = 0
    distv = distance.pdist(x_data_1, metric='correlation')

    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    # Get affinity from similarity matrix
    sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
    # plt.matshow(sparse_graph)
    # plt.savefig('features_sparsegraph.png', bbox_inches='tight')
    # exit()
    graph = Reader.get_affinity(sparse_graph, idx)

    x_data = features.astype(float)#np.identity(num_nodes)
    xrow,xcol = np.shape(x_data)
    for i in range(xrow):
        for j in range(xcol):
            x_data[i, j] = round(x_data[i,j], 4)
    np.savetxt("x_data.csv", x_data, delimiter=',')
    x_data[np.where(np.isnan(x_data))] = 0
    print(np.where(np.isnan(x_data)))
    #exit()
    # Linear classifier
    clf = RidgeClassifier()
    clf.fit(x_data[train_ind, :], y[train_ind].ravel())
    # Compute the accuracy
    lin_acc = clf.score(x_data[test_ind, :], y[test_ind].ravel())
    # Compute the AUC
    pred = clf.decision_function(x_data[test_ind, :])

    y_one_hot = label_binarize(y[test_ind], classes=np.arange(3))
    lin_auc = sklearn.metrics.roc_auc_score(y_one_hot, pred)

    # np.savetxt("x_data.csv", x_data, delimiter = ',')
    # Classification with GCNs
    test_acc, test_auc, weights, confusion = Train.run_training(graph, sparse.coo_matrix(x_data).tolil(), y_data, train_ind, val_ind,
                                            test_ind, params, pathToSave, i)

    # print(test_acc)
    scores_acc = [test_acc]
    scores_auc = [test_auc]
    scores_lin = [lin_acc]
    scores_auc_lin = [lin_auc]
    fold_size = [fold_size]
    if FLAGS.model == 'gcn_cheby':
        weights_0 = weights[0]
        weights_1 = weights[1]
        weights_2 = weights[2]

    scores_lin_ = np.sum(scores_lin)
    scores_auc_lin_ = np.mean(scores_auc_lin)
    scores_acc_ = int(np.sum(scores_acc) * len(test_ind))
    scores_auc_ = np.mean(scores_auc)

    if not os.path.exists(pathToSave + 'excel/'):
        os.makedirs(pathToSave + 'excel/')
    pathToSave2 = pathToSave + 'excel/'
    result_name = 'ABIDE_classification.mat'
    if FLAGS.model == 'gcn_cheby':
        sio.savemat(pathToSave2 + str(trial) + result_name,
                    {'lin': scores_lin_, 'lin_auc': scores_auc_lin_,
                     'acc': scores_acc_, 'auc': scores_auc_, 'folds': num_nodes, 'weights_0': weights_0,
                     'weights_1': weights_1, 'weights_2': weights_2})
        df = pd.DataFrame({'scores_acc': [scores_acc_], 'scores_auc': [scores_auc_], 'scores_lin': [scores_lin_],
                           'scores_auc_lin': [scores_auc_lin_], 'weights_0': weights_0, 'weights_1': weights_1,
                           'weights_2':weights_2, 'confusion_matrix': [confusion]})
    else:
        sio.savemat(pathToSave2 + str(trial) + result_name,
                    {'lin': scores_lin_, 'lin_auc': scores_auc_lin_,
                     'acc': scores_acc_, 'auc': scores_auc_, 'folds': num_nodes})
        df = pd.DataFrame({'scores_acc': [scores_acc_], 'scores_auc': [scores_auc_], 'scores_lin': [scores_lin_],
                           'scores_auc_lin': [scores_auc_lin_], 'confusion_matrix': [confusion]})

    prediction.append(df)

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer_n = pd.ExcelWriter(pathToSave2 + str(test_ind[0]) + '.xlsx', engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer_n, sheet_name='Sheet1')
    # Close the Pandas Excel writer and output the Excel file.
    writer_n.save()

    lin_acc = int(round(lin_acc * len(test_ind)))
    scores_acc = [test_acc]
    scores_auc = [test_auc]
    scores_lin = [lin_acc]
    scores_auc_lin = [lin_auc]
    fold_size = [fold_size]

    # return number of correctly classified samples instead of percentage
    test_acc = int(round(test_acc * len(test_ind)))
    return test_acc, test_auc, lin_acc, lin_auc, fold_size, len(test_ind)


def main(idx, lr, trial, prediction, dp):
    # parameters
    dropout = dp
    decay = 5e-4
    hidden = 16
    hidden2 = 16
    lrate = 0.00005
    lrate2 = idx
    epochs = lr
    num_features = 2000
    parser = argparse.ArgumentParser(description='Graph CNNs for population graphs: '
                                                 'classification of the ABIDE dataset')
    parser.add_argument('--dropout', default=dropout, type=float,
                        help='Dropout rate (1 - keep probability) (default: 0.3)')
    parser.add_argument('--decay', default=decay, type=float,
                        help='Weight for L2 loss on embedding matrix (default: 5e-4)')
    parser.add_argument('--hidden', default=hidden, type=int, help='Number of filters in hidden layers (default: 16)')
    parser.add_argument('--hidden2', default=hidden2, type=int, help='Number of filters in hidden layers (default: 16)')
    parser.add_argument('--lrate', default=lrate, type=float, help='Initial learning rate (default: 0.005)')
    parser.add_argument('--lrate2', default=lrate2, type=float, help='Initial learning rate (default: 0.005)')
    parser.add_argument('--atlas', default='ho', help='atlas for network construction (node definition) (default: ho, '
                                                      'see preprocessed-connectomes-project.org/abide/Pipelines.html '
                                                      'for more options )')
    parser.add_argument('--epochs', default=epochs, type=int, help='Number of epochs to train')
    parser.add_argument('--num_features', default=num_features, type=int, help='Number of features to keep for '
                                                                               'the feature selection step (default: 2000)')
    parser.add_argument('--num_training', default=1.0, type=float, help='Percentage of training set used for '
                                                                        'training (default: 1.0)')
    parser.add_argument('--depth', default=0, type=int, help='Number of additional hidden layers in the GCN. '
                                                             'Total number of hidden layers: 1+depth (default: 0)')
    parser.add_argument('--model', default='dense', help='gcn model used (default: gcn_cheby, '
                                                             'uses chebyshev polynomials, '
                                                             'options: gcn, gcn_cheby, dense )')
    parser.add_argument('--seed', default=5, type=int, help='Seed for random initialisation (default: 123)')  # !--
    parser.add_argument('--folds', default=11, type=int,
                        help='For cross validation, specifies which fold will be '  # !--
                             'used. All folds are used if set to 11 (default: 11)')
    parser.add_argument('--save', default=1, type=int, help='Parameter that specifies if results have to be saved. '
                                                            'Results will be saved if set to 1 (default: 1)')
    parser.add_argument('--connectivity', default='correlation', help='Type of connectivity used for network '
                                                                      'construction (default: correlation, '
                                                                      'options: correlation, partial correlation, '
                                                                      'tangent)')
    parser.add_argument('--branches', default=3, help='Number of parallel branches')
    args = parser.parse_args()
    start_time = time.time()

    # GCN Parameters
    params = dict()
    params['model'] = args.model  # gcn model using chebyshev polynomials
    params['lrate'] = args.lrate  # Initial learning rate
    params['epochs'] = args.epochs  # Number of epochs to train
    params['dropout'] = args.dropout  # Dropout rate (1 - keep probability)
    params['hidden'] = args.hidden  # Number of units in hidden layers
    params['decay'] = args.decay  # Weight for L2 loss on embedding matrix.
    params['early_stopping'] = params[
        'epochs']  # Tolerance for early stopping (# of epochs). No early stopping if set to param.epochs
    params['max_degree'] = 3  # Maximum Chebyshev polynomial degree.
    params[
        'depth'] = args.depth  # number of additional hidden layers in the GCN. Total number of hidden layers: 1+depth
    params['seed'] = args.seed  # seed for random initialisation

    # GCN Parameters
    params['num_features'] = args.num_features  # number of features for feature selection step
    params['num_training'] = args.num_training  # percentage of training set used for training
    atlas = args.atlas  # atlas for network construction (node definition)
    connectivity = args.connectivity  # type of connectivity used for network construction
    params['branches'] = args.branches

    # Get class labels
    # subject_IDs = Reader.get_ids()
    # labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')

    # Get acquisition site
    # sites = Reader.get_subject_score(subject_IDs, score='SITE_ID')
    # unique = np.unique(list(sites.values())).tolist()

    num_classes = Reader.num_classes  # !--
    # num_nodes = Reader.num_nodes  # !--len(subject_IDs)
    # Initialise variables for class labels and acquisition sites
    # y = np.zeros([num_nodes, 1])
    subject_labels, features, all_idx = Reader.get_features()
    num_nodes = all_idx.shape[0]
    y_data = np.zeros([num_nodes, num_classes])
    #features = np.identity(num_nodes)#np.zeros((num_nodes, num_nodes))
    y = subject_labels
    y = y.astype(int)
    # Get class labels and acquisition site for all subjects
    for i in range(num_nodes):
        y_data[i, y[i]-1] = 1
        y[i] = y[i] -1
    splits = 10
    skf = StratifiedKFold(n_splits=splits)
   # pathToSave = './CrossValidation/TrainableTrue/' + str(lr) + '/habit_color_random_feature_with_sparsegraph' + str(trial) + '/'
    pathToSave = './CrossValidation/Shayan_crossEntropy/Balanced_apoe_age_gender/TrainableFalse2/' + str(lr) + '/apoe_gender_age_constant_0.005' + str(trial) + '/'

    subject_IDs = np.zeros(num_nodes)
    graph = np.zeros((num_nodes, num_nodes))
    if args.folds == 11:  # run cross validation on all folds
        # scores = Parallel(n_jobs=splits)(delayed(train_fold)(train_ind, test_ind, test_ind, graph, features, y, y_data,
        #                                                  params, subject_IDs, pathToSave, i)
        #                              for train_ind, test_ind in
        #                              reversed(list(skf.split(np.zeros(num_nodes), np.squeeze(y)))))
        scores = [train_fold(train_ind, test_ind, test_ind, graph, features, y, y_data,
                             params, subject_IDs, pathToSave, i, subject_labels, all_idx) for train_ind, test_ind in
                  reversed(list(skf.split(np.zeros(num_nodes), np.squeeze(y))))]
        string = ''
        print(scores)
        string += str(scores)
        scores_acc = [x[0] for x in scores]
        scores_auc = [x[1] for x in scores]
        scores_lin = [x[2] for x in scores]
        scores_auc_lin = [x[3] for x in scores]
        fold_size = [x[4] for x in scores]
        test_ind = [x[5] for x in scores]
        print("test indices ", np.sum(test_ind))
        print('overall linear accuracy %f' + str(np.sum(scores_lin) * 1. / np.sum(test_ind)))  # !--
        print('overall linear AUC %f' + str(np.mean(scores_auc_lin)))
        print('overall accuracy %f' + str(np.sum(scores_acc) * 1. / np.sum(test_ind)))  # !--
        print('overall AUC %f' + str(np.mean(scores_auc)))

        string +='\n'
        string += 'overall linear accuracy ' + str(np.sum(scores_lin) * 1. / np.sum(test_ind))
        string += 'overall linear AUC ' + str(np.mean(scores_auc_lin))
        string += 'overall accuracy ' + str(np.sum(scores_acc) * 1. / np.sum(test_ind))
        string += 'overall AUC ' + str(np.mean(scores_auc))

        file = open(pathToSave + '/excel/' + "print_results.txt" , 'w')
        file.write(string)
        file.close()
        scores_lin_ = np.sum(scores_lin) * 1. / num_nodes
        scores_auc_lin_ = np.mean(scores_auc_lin)
        scores_acc_ = np.sum(scores_acc) * 1. / num_nodes
        scores_auc_ = np.mean(scores_auc)

        result_name = 'ABIDE_classification'
        sio.savemat(pathToSave + result_name,
                    {'lin': scores_lin_, 'lin_auc': scores_auc_lin_,
                     'acc': scores_acc_, 'auc': scores_auc_, 'folds': fold_size})
        sio.savemat(pathToSave + 'param',
                    {'dropout': dropout, 'decay': decay,
                     'hidden': hidden, 'hidden2': hidden2, 'lrate': lrate, 'epochs': epochs,
                     'num_features': num_features})

        df = pd.DataFrame({'scores_acc': [scores_acc_], 'scores_auc': [scores_auc_], 'scores_lin': [scores_lin_],
                           'scores_auc_lin': [scores_auc_lin_]})

        prediction.append(df)
        #
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(pathToSave + '/excel/' + "total_results" + '.xlsx', engine='xlsxwriter')
        # Convert the dataframe to an XlsxWriter Excel object.
        df.to_excel(writer, sheet_name='Sheet1')
        # Close the Pandas Excel writer and output the Excel file.
        writer.save()


    else:  # compute results for only one fold

        cv_splits = list(skf.split(features, np.squeeze(y)))

        train = cv_splits[args.folds][0]
        test = cv_splits[args.folds][1]

        val = test

        scores_acc, scores_auc, scores_lin, scores_auc_lin, fold_size, _ = train_fold(train, test, val, graph, features,
                                                                                      y, y_data, params, subject_IDs,
                                                                                      pathToSave, i,subject_labels,
                                                                                      all_idx)

        print('overall linear accuracy ' + str(np.sum(scores_lin) * 1. / fold_size))
        print('overall linear AUC ' + str(np.mean(scores_auc_lin)))
        print('overall accuracy ' + str(np.sum(scores_acc) * 1. / fold_size))
        print('overall AUC ' + str(np.mean(scores_auc)))

    if args.save == 1:
        result_name = 'ABIDE_classification.mat'
        sio.savemat('./' + result_name + '.mat',
                    {'lin': scores_lin, 'lin_auc': scores_auc_lin,
                     'acc': scores_acc, 'auc': scores_auc, 'folds': fold_size})
    duration = (time.time() - start_time) / 60
    print("time in minutes:", duration)


if __name__ == "__main__":

    Lr = [500]
    idx2 = [0.00001, 0.0001]
    dropout_ = [0.3]
    prediction = pd.DataFrame({'scores_acc': [0], 'scores_auc': [0], 'scores_lin': [0], 'scores_auc_lin': [0]})
    for j in range(0, 1):
        for i in range(0, 1):
            for drop in range(0, 1):
                for trial in range(0, 1):
                    tf.reset_default_graph()
                    tf.app.flags._global_parser = argparse.ArgumentParser()
                    # idx = fraction[i]
                    lr = Lr[j]  # epochs
                    idx = idx2[i]  # learning rate
                    dp = dropout_[drop]
                    parser = argparse.ArgumentParser(description='To pass iteration')
                    parser.add_argument('--idx', default=idx)
                    parser.add_argument('--lr', default=lr)
                    parser.add_argument('--trial', default=trial)
                    parser.add_argument('--dp', default=dp)
                    parser.add_argument('--prediction', default=prediction)
                    args = parser.parse_args()

                    main(idx=args.idx, lr=args.lr, trial=args.trial, prediction=args.prediction, dp=args.dp)