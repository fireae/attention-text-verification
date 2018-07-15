"""Various implementations of sequence layers for character prediction.

A 'sequence layer' is a part of a computation graph which is responsible of
producing a sequence of characters using extracted image features. There are
many reasonable ways to implement such layers. All of them are using RNNs.
This module provides implementations which uses 'attention' mechanism to
spatially 'pool' image features and also can use a previously predicted
character to predict the next (aka auto regression).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import abc
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim


def orthogonal_initializer(shape, dtype = tf.float32, *args, **kwargs):
    """Generates orthonormal matrices with random values.

    Orthonormal initialization is important for RNNs:
      http://arxiv.org/abs/1312.6120
      http://smerity.com/articles/2016/orthogonal_init.html

    For non-square shapes the returned matrix will be semi-orthonormal: if the
    number of columns exceeds the number of rows, then the rows are orthonormal
    vectors; but if the number of rows exceeds the number of columns, then the
    columns are orthonormal vectors.

    We use SVD decomposition to generate an orthonormal matrix with random
    values. The same way as it is done in the Lasagne library for Theano. Note
    that both u and v returned by the svd are orthogonal and random. We just need
    to pick one with the right shape.

    Args:
      shape: a shape of the tensor matrix to initialize.
      dtype: a dtype of the initialized tensor.
      *args: not used.
      **kwargs: not used.

    Returns:
      An initialized tensor.
    """
    del args
    del kwargs
    flat_shape = (shape[0], np.prod(shape[1:]))
    w = np.random.randn(*flat_shape)
    u, _, v = np.linalg.svd(w, full_matrices = False)
    w = u if u.shape == flat_shape else v
    return tf.constant(w.reshape(shape), dtype = dtype)


SequenceLayerParams = collections.namedtuple('SequenceLogitsParams', [
    'num_lstm_units', 'weight_decay', 'lstm_state_clip_value'
])


class SequenceLayer(object):
    """A class for sequence layers.
    """
    __metaclass__ = abc.ABCMeta


    def __init__(self, net, labels, model_params, method_params):
        """Stores argument in member variable for further use.

        Args:
          net: A tensor with shape [batch_size, num_features, feature_size] which
            contains some extracted image features.
          labels_one_hot: An optional (can be None) ground truth labels for the
            input features. Is a tensor with shape
            [batch_size, seq_length, num_char_classes]
          model_params: A namedtuple with model parameters (model.ModelParams).
          method_params: A SequenceLayerParams instance.
        """
        self._params = model_params
        self._mparams = method_params
        self._batch_size = net.get_shape().dims[0].value
        self._labels_one_hot = tf.one_hot(indices = labels, depth = self._params.num_char_classes, axis = 2)
        self._net = net

    def get_input(self, prev, i):
        return self._labels_one_hot[:, i, :]

    def create_logits(self):
        """Creates character sequence logits for a net specified in the constructor.

        A "main" method for the sequence layer which glues together all pieces.

        Returns:
          A tensor with shape [batch_size].
        """
        with tf.variable_scope('LSTM'):
            decoder_inputs = [self.get_input(prev = None, i = 0)] + [None] * (self._params.seq_length - 1)
            lstm_cell = tf.contrib.rnn.LSTMCell(
                    self._mparams.num_lstm_units,
                    use_peepholes = False,
                    cell_clip = self._mparams.lstm_state_clip_value,
                    state_is_tuple = True,
                    initializer = orthogonal_initializer)
            lstm_outputs, _ = tf.contrib.legacy_seq2seq.attention_decoder(
                    decoder_inputs = decoder_inputs,
                    initial_state = lstm_cell.zero_state(self._batch_size, tf.float32),
                    attention_states = self._net,
                    cell = lstm_cell,
                    loop_function = self.get_input)

        return tf.nn.softmax(lstm_outputs[-1])
