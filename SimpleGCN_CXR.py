
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


import time
import argparse
import os

import numpy as np
from scipy import sparse
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from scipy.spatial import distance
from sklearn.linear_model import RidgeClassifier
import sklearn.metrics
import scipy.io as sio

import ABIDEParser as Reader
import gcn.train_GCNSimpleGCN as Train
import pandas as pd
import tensorflow as tf
import random
import read_csv as reader


# Prepares the training/test data for each cross validation fold and trains the GCN
def train_fold(train_ind, test_ind, val_ind, graph_feat, features, y, y_data, params, subject_IDs, pathToSave):
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
    tf.reset_default_graph()
    tf.app.flags._global_parser = argparse.ArgumentParser()
    print(len(train_ind))

    # selection of a subset of data if running experiments with a subset of the training set
    # = Reader.site_percentage(train_ind, params['num_training'], subject_IDs)
    labeled_ind = reader.site_percentage(train_ind, 1.0)
    # feature selection/dimensionality reduction step
    x_data = Reader.feature_selection(features, y, labeled_ind, params['num_features'])

    fold_size = len(test_ind)

    # Calculate all pairwise distances
    distv = distance.pdist(x_data, metric='correlation')
    # Convert to a square symmetric distance matrix
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    # Get affinity from similarity matrix
    sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
    num_nodes = 662
    Randgraph = np.zeros((num_nodes, num_nodes))
    for k in range(num_nodes):
        for j in range(k + 1, num_nodes):
            try:
                val = random.uniform(0, 1)
                if val < 0.5:
                    Randgraph[k, j] += 1
                    Randgraph[j, k] += 1
            except ValueError:  # missing label
                pass

    #graph_feat1 = graph_feat + Randgraph
    final_graph = graph_feat * sparse_graph
    #sio.savemat('/media/lesliec/Volume/Anees/ParallelGCN/ReproduceBestsofar31012018/2feb_after_meeting/7feb_After_Sitepackagebug/OriginalGCN' + str(trial) + 'SITE_graph',
    #            {'final_graph': final_graph, 'sparse_graph': sparse_graph,'graph_feat': graph_feat  })

    # Linear classifier
    clf = RidgeClassifier()
    clf.fit(x_data[train_ind, :], y[train_ind].ravel())
    # Compute the accuracy
    lin_acc = clf.score(x_data[test_ind, :], y[test_ind].ravel())
    # Compute the AUC
    pred = clf.decision_function(x_data[test_ind, :])
    lin_auc = sklearn.metrics.roc_auc_score(y[test_ind] - 1, pred)

    print("Linear Accuracy: " + str(lin_acc))

    # Classification with GCNs
    test_acc, test_auc = Train.run_training(final_graph, sparse.coo_matrix(x_data).tolil(), y_data, train_ind, val_ind,
                                            test_ind, params, pathToSave)

    # return number of correctly classified samples instead of percentage
    # test_acc = int(round(test_acc * len(test_ind)))
    # lin_acc = int(round(lin_acc * len(test_ind)))
    scores_acc = [test_acc]
    scores_auc = [test_auc]
    scores_lin = [lin_acc]
    scores_auc_lin = [lin_auc]
    fold_size = [fold_size]
    weights_0 = 0
    weights_1 = 0

    scores_lin_ = np.sum(scores_lin)
    scores_auc_lin_ = np.mean(scores_auc_lin)
    scores_acc_ = np.sum(scores_acc)
    scores_auc_ = np.mean(scores_auc)

    if not os.path.exists(pathToSave + 'excel/'):
        os.makedirs(pathToSave + 'excel/')
    pathToSave2 = pathToSave + 'excel/'
    result_name = 'ABIDE_classification.mat'
    sio.savemat(pathToSave2 + str(trial) + result_name,
                {'lin': scores_lin_, 'lin_auc': scores_auc_lin_,
                 'acc': scores_acc_, 'auc': scores_auc_, 'folds': num_nodes, 'weights_0': weights_0, 'weights_1': weights_1})
    df = pd.DataFrame({'scores_acc': [scores_acc_], 'scores_auc': [scores_auc_], 'scores_lin': [scores_lin_],
                       'scores_auc_lin': [scores_auc_lin_], 'weights_0': weights_0, 'weights_1': weights_1})

    prediction.append(df)

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer_n = pd.ExcelWriter(pathToSave2 + str(test_ind[0]) + '.xlsx', engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer_n, sheet_name='Sheet1')
    # Close the Pandas Excel writer and output the Excel file.
    writer_n.save()

    test_acc = int(round(test_acc * len(test_ind)))
    lin_acc = int(round(lin_acc * len(test_ind)))
    scores_acc = [test_acc]
    scores_auc = [test_auc]
    scores_lin = [lin_acc]
    scores_auc_lin = [lin_auc]
    fold_size = [fold_size]

    # return number of correctly classified samples instead of percentage
    #test_acc = int(round(test_acc * len(test_ind)))
    return test_acc, test_auc, lin_acc, lin_auc, fold_size

def main(idx, lr, trial, prediction):
    epochs = lr
    parser = argparse.ArgumentParser(description='Graph CNNs for population graphs: '
                                                 'classification of the ABIDE dataset')
    parser.add_argument('--dropout', default=0.3, type=float,
                        help='Dropout rate (1 - keep probability) (default: 0.3)')
    parser.add_argument('--decay', default=5e-4, type=float,
                        help='Weight for L2 loss on embedding matrix (default: 5e-4)')
    parser.add_argument('--hidden', default=16, type=int, help='Number of filters in hidden layers (default: 16)')
    parser.add_argument('--lrate', default=0.005, type=float, help='Initial learning rate (default: 0.005)')
    parser.add_argument('--atlas', default='ho', help='atlas for network construction (node definition) (default: ho, '
                                                      'see preprocessed-connectomes-project.org/abide/Pipelines.html '
                                                      'for more options )')
    parser.add_argument('--epochs', default=epochs, type=int, help='Number of epochs to train')
    parser.add_argument('--num_features', default=2000, type=int, help='Number of features to keep for '
                                                                       'the feature selection step (default: 2000)')
    parser.add_argument('--num_training', default=1.0, type=float, help='Percentage of training set used for '
                                                                        'training (default: 1.0)')
    parser.add_argument('--depth', default=0, type=int, help='Number of additional hidden layers in the GCN. '
                                                             'Total number of hidden layers: 1+depth (default: 0)')
    parser.add_argument('--model', default='gcn_cheby', help='gcn model used (default: gcn_cheby, '
                                                             'uses chebyshev polynomials, '
                                                             'options: gcn, gcn_cheby, dense )')
    parser.add_argument('--seed', default=123, type=int, help='Seed for random initialisation (default: 123)')
    parser.add_argument('--folds', default=11, type=int, help='For cross validation, specifies which fold will be '
                                                             'used. All folds are used if set to 11 (default: 11)')
    parser.add_argument('--save', default=1, type=int, help='Parameter that specifies if results have to be saved. '
                                                            'Results will be saved if set to 1 (default: 1)')
    parser.add_argument('--connectivity', default='correlation', help='Type of connectivity used for network '
                                                                      'construction (default: correlation, '
                                                                      'options: correlation, partial correlation, '
                                                                      'tangent)')

    args = parser.parse_args()
    start_time = time.time()

    # GCN Parameters
    params = dict()
    params['model'] = args.model                    # gcn model using chebyshev polynomials
    params['lrate'] = args.lrate                    # Initial learning rate
    params['epochs'] = args.epochs                  # Number of epochs to train
    params['dropout'] = args.dropout                # Dropout rate (1 - keep probability)
    params['hidden'] = args.hidden                  # Number of units in hidden layers
    params['decay'] = args.decay                    # Weight for L2 loss on embedding matrix.
    params['early_stopping'] = params['epochs']     # Tolerance for early stopping (# of epochs). No early stopping if set to param.epochs
    params['max_degree'] = 3                        # Maximum Chebyshev polynomial degree.
    params['depth'] = args.depth                    # number of additional hidden layers in the GCN. Total number of hidden layers: 1+depth
    params['seed'] = args.seed                      # seed for random initialisation

    # GCN Parameters
    params['num_features'] = args.num_features      # number of features for feature selection step
    params['num_training'] = args.num_training      # percentage of training set used for training
    atlas = args.atlas                              # atlas for network construction (node definition)
    connectivity = args.connectivity                # type of connectivity used for network construction

    # Get class labels

    # Get acquisition site

    num_classes = 2
    num_nodes = 662

    # Initialise variables for class labels and acquisition sites


    # Compute feature vectors (vectorised connectivity networks)

    # Compute population graph using gender and acquisition site
    #graph = Reader.create_affinity_graph_from_scores(['SEX', 'SITE_ID'], subject_IDs)
    features = reader.get_ChestXray()
    y_data, y, labels = reader.get_labelChestXray()
    num_nodes = 662
    graph1 = reader.get_genderGraphAge2()
    graph2 = reader.get_genderGraphGen()

    w1 = 1.0 #Gender
    w2 = 1.0 #Age
    g1 = (w1 * graph1) #Age
    g2  = (w2 * graph2) # Gender
    graph = g1 + g2

    # Folds for cross validation experiments
    skf = StratifiedKFold(n_splits=10)
    subject_IDs = 1

    pathToSave = '/home/leslie/Downloads/Parallel2br_chestxray-master/CrossValidation/BaseLine_AgeGender_11_expDecay_Shuffle_TRY' + str(trial) + '/'

    if args.folds == 11:  # run cross validation on all folds
        # scores = Parallel(n_jobs=5)(delayed(train_fold)(train_ind, test_ind, test_ind, graph, features, y, y_data,
        #                                                  params, subject_IDs,pathToSave)
        #                              for train_ind, test_ind in
        #                              reversed(list(skf.split(np.zeros(num_nodes), np.squeeze(y)))))

        scores = [train_fold(train_ind, test_ind, test_ind, graph, features, y, y_data,params, subject_IDs,pathToSave)
                                      for train_ind, test_ind in reversed(list(skf.split(np.zeros(num_nodes), np.squeeze(y))))]
        print(scores)

        scores_acc = [x[0] for x in scores]
        scores_auc = [x[1] for x in scores]
        scores_lin = [x[2] for x in scores]
        scores_auc_lin = [x[3] for x in scores]
        fold_size = [x[4] for x in scores]

        print('overall linear accuracy %f' + str(np.sum(scores_lin) * 1. / num_nodes))
        print('overall linear AUC %f' + str(np.mean(scores_auc_lin)))
        print('overall accuracy %f' + str(np.sum(scores_acc) * 1. / num_nodes))
        print('overall AUC %f' + str(np.mean(scores_auc)))

        scores_lin_ = np.sum(scores_lin) * 1. / num_nodes
        scores_auc_lin_ = np.mean(scores_auc_lin)
        scores_acc_ = np.sum(scores_acc) * 1. / num_nodes
        scores_auc_ = np.mean(scores_auc)

        if args.save == 1:
            result_name = 'ABIDE_classification.mat'
            sio.savemat(pathToSave + str(trial) + result_name,
                        {'lin': scores_lin_, 'lin_auc': scores_auc_lin_,
                        'acc': scores_acc_, 'auc': scores_auc_, 'folds': num_nodes})
            df = pd.DataFrame({'scores_acc': [scores_acc_], 'scores_auc': [scores_auc_], 'scores_lin': [scores_lin_],
                               'scores_auc_lin': [scores_auc_lin_]})

            prediction.append(df)

            # Create a Pandas Excel writer using XlsxWriter as the engine.
            writer_n = pd.ExcelWriter(pathToSave + str(trial) + '.xlsx', engine='xlsxwriter')
            # Convert the dataframe to an XlsxWriter Excel object.
            df.to_excel(writer_n, sheet_name='Sheet1')
            # Close the Pandas Excel writer and output the Excel file.
            writer_n.save()

    else:  # compute results for only one fold

        cv_splits = list(skf.split(features, np.squeeze(y)))

        train = cv_splits[args.folds][0]
        test = cv_splits[args.folds][1]

        val = test

        scores_acc, scores_auc, scores_lin, scores_auc_lin, fold_size = train_fold(train, test, val, graph, features, y,
                                                         y_data, params, subject_IDs, pathToSave)

        print('overall linear accuracy %f' + str(np.sum(scores_lin) * 1. / fold_size))
        print('overall linear AUC %f' + str(np.mean(scores_auc_lin)))
        print('overall accuracy %f' + str(np.sum(scores_acc) * 1. / fold_size))
        print('overall AUC %f' + str(np.mean(scores_auc)))

        scores_lin_ = np.sum(scores_lin) * 1. / fold_size
        scores_auc_lin_ = np.mean(scores_auc_lin)
        scores_acc_ = np.sum(scores_acc) * 1. / fold_size
        scores_auc_ = np.mean(scores_auc)
    if args.save == 1:
        result_name = 'ABIDE_classification.mat'
        sio.savemat(pathToSave + str(trial) + result_name,
                    {'lin': scores_lin_, 'lin_auc': scores_auc_lin_,
                     'acc': scores_acc_, 'auc': scores_auc_, 'folds': fold_size})
        df = pd.DataFrame({'scores_acc': [scores_acc_], 'scores_auc': [scores_auc_], 'scores_lin': [scores_lin_],
                           'scores_auc_lin': [scores_auc_lin_]})

        prediction.append(df)

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer_n = pd.ExcelWriter(pathToSave + str(trial) + '.xlsx', engine='xlsxwriter')
        # Convert the dataframe to an XlsxWriter Excel object.
        df.to_excel(writer_n, sheet_name='Sheet1')
        # Close the Pandas Excel writer and output the Excel file.
        writer_n.save()

    duration = (time.time() - start_time)/60
    print("time in minutes:",duration)
if __name__ == "__main__":
    #idx1= 110
    fraction = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    #Lr = [150, 250]
    Lr = [150]
    prediction = pd.DataFrame({'scores_acc': [0], 'scores_auc': [0], 'scores_lin':[0], 'scores_auc_lin': [0]})
    for j in range(0, 1):
        for trial in range(0, 1):
            for i in range(0, 1):
                tf.reset_default_graph()
                tf.app.flags._global_parser = argparse.ArgumentParser()
                idx = fraction[i]
                lr = Lr[j]
                #idx=idx/100
                parser = argparse.ArgumentParser(description='To pass iteration')
                parser.add_argument('--idx', default=idx)
                parser.add_argument('--lr', default=lr)
                parser.add_argument('--trial', default=trial)
                parser.add_argument('--prediction', default=prediction)
                args = parser.parse_args()

                main(idx=args.idx, lr=args.lr, trial=args.trial, prediction=args.prediction)

