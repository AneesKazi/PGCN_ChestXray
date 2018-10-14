# Copyright (c) 2016 Thomas Kipf
# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold

import random
from gcn.utils import *
from gcn.models import MLP, Deep_GCN, ParallelGCN
import sklearn.metrics
import matplotlib.pyplot as plt
import datetime

def get_train_test_masks(labels, idx_train, idx_val, idx_test):
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return y_train, y_val, y_test, train_mask, val_mask, test_mask


def run_training(adjs1, adjs2, features, labels, idx_train, idx_val, idx_test, idx, lr,
                 params, pathToSave, i):
    # Set random seed
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    tf.set_random_seed(params['seed'])
    now = datetime.datetime.now()
    pathToSave += str(now.hour) + "_" + str(now.minute)

    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('model', params['model'], 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_float('learning_rate', params['lrate'], 'Initial learning rate.')
    flags.DEFINE_integer('epochs', params['epochs'], 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', params['hidden'], 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', params['dropout'], 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', params['decay'], 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', params['early_stopping'], 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', params['max_degree'], 'Maximum Chebyshev polynomial degree.')
    flags.DEFINE_integer('depth', params['depth'], 'Depth of Deep GCN')

    # Create test, val and train masked variables
    y_train, y_val, y_test, train_mask, val_mask, test_mask = get_train_test_masks(labels, idx_train, idx_val, idx_test)

    # Some preprocessing
    features = preprocess_features(features)
    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = Deep_GCN
    elif FLAGS.model == 'gcn_cheby':
        support0 = chebyshev_polynomials(adjs1, FLAGS.max_degree)
        support1 = chebyshev_polynomials(adjs2, FLAGS.max_degree)

        #support3 = chebyshev_polynomials(adjs[3], FLAGS.max_degree)
        #print(support, "support")
        num_supports = 1 + FLAGS.max_degree
        model_func = ParallelGCN
    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for GCN model ')

    # Define placeholders
    placeholders = {
        'support0': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'support1': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        #'support3': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'phase_train': tf.placeholder_with_default(False, shape=()),
        'branches': tf.placeholder_with_default(4, shape=()),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = model_func(placeholders, input_dim=features[2][1], logging=True)
    weights111 = model.variables_3
    # Initialize session
    sess = tf.Session()
    tf.summary.scalar("accuracy", model.accuracy)
    tf.summary.scalar("loss", model.loss)
    # tf.summary.tensor("predict", model.predict())
    merged_summary = tf.summary.merge_all()
    # writer = tf.summary.FileWriter('/home/aneescamp/cnn_graph/GCN_Sarah/population-gcn-master/simpleGraph' + str(idx) +'/graph/')
    # writer = tf.summary.FileWriter('/media/aneescamp/Volume/Anees/SimpleGCN2/'+ str(idx))

    writer = tf.summary.FileWriter(pathToSave + 'tr/')
    writer.add_graph(sess.graph)
    writer2 = tf.summary.FileWriter(pathToSave + 'te/')
    writer2.add_graph(sess.graph)

    # Define model evaluation function
    def evaluate(feats, graph0, graph1, label, mask, placeholder):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(feats, graph0, graph1, label, mask, placeholder)
        feed_dict_val.update({placeholder['phase_train'].name: False})
        s2 = sess.run(merged_summary, feed_dict=feed_dict_val)
        # print(s)
        writer2.add_summary(s2, epoch)
        model.variables_3 = model.variables_3 / tf.reduce_sum(model.variables_3)

        outs_val = sess.run([model.loss, model.accuracy, model.predict()], feed_dict=feed_dict_val)

        # Compute the area under curve
        pred = outs_val[2]
        pred = pred[np.squeeze(np.argwhere(mask == 1)), :]
        lab = label
        lab = lab[np.squeeze(np.argwhere(mask == 1)), :]
        auc = sklearn.metrics.roc_auc_score(np.squeeze(lab), np.squeeze(pred))
        return outs_val[0], outs_val[1], auc, (time.time() - t_test)

    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []
    weightsOut0 = []
    weightsOut1 = []
    weightsOut2 = []
    #weightsOut3 = []
    validation_accuracies = []
    train_accuracies = []
    x_range = []
    # Train model
    for epoch in range(params['epochs']):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support0, support1, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout, placeholders['phase_train']: True})
        s = sess.run(merged_summary, feed_dict=feed_dict)
        writer.add_summary(s, epoch)

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy, model.predict()], feed_dict=feed_dict)
        if epoch % 50 == 0 or epoch > 150:
            model.variables_3 = model.variables_3 / tf.reduce_sum(model.variables_3)
            weights = sess.run(model.variables_3)
            print(weights, "the weights are")

        # sub_folder = "./three_no_feature_sparsegraph_acti/"
        # if epoch == params['epochs'] - 1:
        #     outputs = sess.run(model.outputs, feed_dict=feed_dict)
        #
        #     label_0 = [i for i, x in enumerate(np.argmax(y_val + y_train, axis=1)) if x == 0]
        #     label_1 = [i for i, x in enumerate(np.argmax(y_val + y_train, axis=1)) if x == 1]
        #
        #     tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        #     Y = tsne.fit_transform(outputs)
        #     fig = plt.figure(15)
        #     ax = fig.add_subplot(111, projection='3d')
        #     plt.scatter(Y[label_0, 0], Y[label_0, 1], c='r', cmap=plt.cm.Spectral)
        #     plt.scatter(Y[label_1, 0], Y[label_1, 1], c='b', cmap=plt.cm.Spectral)
        #     plt.axis('tight')
        #     plt.savefig(sub_folder + 'activation_2.png', bbox_inches='tight')
        #     plt.show()
        #     exit()
        pred = outs[3]
        #print("pred",pred)
        pred = pred[np.squeeze(np.argwhere(train_mask == 1)), :]
        labs = y_train
        labs = labs[np.squeeze(np.argwhere(train_mask == 1)), :]
        train_auc = sklearn.metrics.roc_auc_score(np.squeeze(labs), np.squeeze(pred))

        # Validation
        cost, acc, auc, duration = evaluate(features, support0, support1, y_val, val_mask, placeholders)
        cost_val.append(cost)

        prediction = sess.run(weights111, feed_dict=feed_dict)
        prediction = np.asarray(prediction)
        # print(prediction, "the weights are ")
        weights1 = prediction
        # #weights1.mean(axis=0)
        weightsOut0.append(weights1[0])
        weightsOut1.append(weights1[1])

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "train_auc=", "{:.5f}".format(train_auc), "val_loss=",
              "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "val_auc=", "{:.5f}".format(auc), "time=",
              "{:.5f}".format(time.time() - t + duration))

        validation_accuracies.append(acc)
        train_accuracies.append(outs[2])
        x_range.append(epoch + 1)

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")
    # !--
    # plt.plot(x_range, train_accuracies, '-b', label='Training_data')
    # plt.plot(x_range, validation_accuracies, '-g', label='Validation_data')
    # plt.legend(loc='lower right', frameon=False)
    # plt.ylim(ymax=1.0, ymin=0)
    # plt.ylabel('accuracy')1
    # plt.xlabel('epochs')
    #plt.savefig("parisot_train90_10.png", box_inches="tight")
    #plt.show()
    # !--

    # Testing
    sess.run(tf.local_variables_initializer())
    test_cost, test_acc, test_auc, test_duration = evaluate(features, support0, support1, y_test, test_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc),
          "auc=", "{:.5f}".format(test_auc))
    model.variables_3 = model.variables_3 / tf.reduce_sum(model.variables_3)
    weights = sess.run(model.variables_3)
    return test_acc, test_auc, weights
