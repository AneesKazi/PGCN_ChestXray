import tensorflow as tf

from gcn.layers import *
from gcn.metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.auc = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        self.auc = masked_auc(self.outputs, self.placeholders['labels'],
                              self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return self.outputs
        return tf.nn.softmax(self.outputs)


class ParallelGCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', True)
        self.logging = logging
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.vars = {}
        self.placeholders = placeholders
        self.var_ = []

        self.layers0 = []
        self.layers1 = []

        self.activations0 = []
        self.activations1 = []
        self.modalNew = []

        self.branch_outputs = []
        self.inputs = placeholders['features']
#        for i in range(placeholders['branches']):
#            self.branch_outputs[i] = None
        self.outputs = None
        self.outputs0 = []
        self.outputs1 = []
        self.branches = 2#self.placeholders['branches']


        self.loss = 0
        self.accuracy = 0
        self.auc = 0
        self.optimizer = None
        self.opt_op = None

        # # Alternate training
        self.global_step = tf.Variable(0, trainable=False)
        self.boundaries = [150, 300]
        self.values = [0.0000001, 0.10, 0.10]
        #self.rateW = tf.train.piecewise_constant(self.global_step, self.boundaries, self.values)
        #self.rateW = tf.train.piecewise_constant(self.global_step, self.boundaries, self.values)

        self.rateW = tf.train.exponential_decay(0.00001, self.global_step, 1, 1.02, staircase=True)

        self.global_step2 = tf.Variable(0, trainable=False)
        self.boundaries2 = [150, 300]
        self.values2 = [0.005, 0.00001, 0.00001]
        self.rateGCN = tf.train.piecewise_constant(self.global_step2, self.boundaries2, self.values2)

        self.optimizer1 = tf.train.AdamOptimizer(self.rateW)
        self.optimizer2 = tf.train.AdamOptimizer(self.rateGCN)

        self.build()

    def lastlayer(self):
        self.outputs = Dense(input_dim=self.branches,
                             output_dim=self.output_dim,
                             placeholders=self.placeholders,
                             act=lambda x: x,
                             dropout=True,
                             logging=self.logging, name='modalNew')

        return self.outputs

    def _build(self):

        self.layers0.append(GraphConvolution0(input_dim=self.input_dim,
                                             output_dim=FLAGS.hidden1,
                                             placeholders=self.placeholders,
                                             act=tf.nn.relu,
                                             dropout=True,
                                             sparse_inputs=True,
                                             logging=self.logging))

        self.layers0.append(GraphConvolution0(input_dim=FLAGS.hidden1,
                                             output_dim=self.output_dim,
                                             placeholders=self.placeholders,
                                             act=lambda x: x,
                                             # act=tf.nn.relu,
                                             dropout=True,
                                             logging=self.logging))

        self.layers1.append(GraphConvolution1(input_dim=self.input_dim,
                                             output_dim=FLAGS.hidden1,
                                             placeholders=self.placeholders,
                                             act=tf.nn.relu,
                                             dropout=True,
                                             sparse_inputs=True,
                                             logging=self.logging))

        self.layers1.append(GraphConvolution1(input_dim=FLAGS.hidden1,
                                             output_dim=self.output_dim,
                                             placeholders=self.placeholders,
                                             act=lambda x: x,
                                             # act=tf.nn.relu,
                                             dropout=True,
                                             logging=self.logging))
        self.modalNew.append(self.lastlayer())

    def build(self):
        with tf.variable_scope(self.name):
            self._build()

        self.activations0.append(self.inputs)
        self.activations1.append(self.inputs)


        for layer in self.layers0:
            hidden = layer(self.activations0[-1])
            self.activations0.append(hidden)
        self.outputs0 = self.activations0[-1]

        for layer in self.layers1:
            hidden = layer(self.activations1[-1])
            self.activations1.append(hidden)
        self.outputs1 = self.activations1[-1]

        self.outputs = self.modalNew[-1](tf.stack([self.outputs0, self.outputs1], axis = 2))

        variables_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.vars = {var.name: var for var in variables_}

        self._loss()
        self._accuracy()

        variables_n = variables_[:-1]
        with tf.variable_scope('model/modalNew_vars', reuse=tf.AUTO_REUSE):
            self.variables_3 = variables_[-1]#tf.get_variable('weights1')
            self.opt_op1 = self.optimizer1.minimize(self.loss, var_list=self.variables_3, global_step=self.global_step)
            self.opt_op2 = self.optimizer2.minimize(self.loss, var_list=variables_n, global_step=self.global_step2)

            self.opt_op = tf.group(self.opt_op1, self.opt_op2)

    def predict(self):
        return tf.nn.softmax(self.outputs)

    def _loss(self):

        for v in self.layers0[0].vars.values():
            self.var_.append(v)
        for v in self.layers1[0].vars.values():
            self.var_.append(v)

        for var in self.var_:
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])


    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _auc(self):
        self.auc = masked_auc(self.outputs, self.placeholders['labels'],
                              self.placeholders['labels_mask'])

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class Deep_GCN(GCN):

    def __init__(self, placeholders, input_dim, depth, **kwargs):
        self.depth = depth
        super(Deep_GCN, self).__init__(placeholders, input_dim, **kwargs)

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        for i in range(self.depth):
            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                                output_dim=FLAGS.hidden1,
                                                placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=True,
                                                logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))
