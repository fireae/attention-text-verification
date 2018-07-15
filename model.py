import sys
import collections
import logging
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import inception
import sequence_layers
import utils


ModelParams = collections.namedtuple('ModelParams', [
    'num_char_classes', 'seq_length', 'num_views', 'null_code'
])

ConvTowerParams = collections.namedtuple('ConvTowerParams', ['final_endpoint'])

SequenceLogitsParams = collections.namedtuple('SequenceLogitsParams', [
    'use_attention', 'use_autoregression', 'num_lstm_units', 'weight_decay',
    'lstm_state_clip_value'
])

SequenceLossParams = collections.namedtuple('SequenceLossParams', [
    'label_smoothing', 'ignore_nulls', 'average_across_timesteps'
])

EncodeCoordinatesParams = collections.namedtuple('EncodeCoordinatesParams', [
    'enabled'
])


def get_softmax_loss_fn(label_smoothing):
    """Returns sparse or dense loss function depending on the label_smoothing.

      Args:
        label_smoothing: weight for label smoothing

      Returns:
        a function which takes labels and predictions as arguments and returns
        a softmax loss for the selected type of labels (sparse or dense).
      """
    if label_smoothing > 0:
        def loss_fn(labels, logits):
            return tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)
    else:
        def loss_fn(labels, logits):
            return tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels)
    return loss_fn


class Model(object):
    """Class to create the Attention OCR Model."""


    def __init__(self,
                 num_char_classes,
                 seq_length,
                 num_views,
                 null_code,
                 mparams = None,
                 charset = None):
        """Initialized model parameters.

        Args:
          num_char_classes: size of character set.
          seq_length: number of characters in a sequence.
          num_views: Number of views (conv towers) to use.
          null_code: A character code corresponding to a character which
            indicates end of a sequence.
          mparams: a dictionary with hyper parameters for methods,  keys -
            function names, values - corresponding namedtuples.
          charset: an optional dictionary with a mapping between character ids and
            utf8 strings. If specified the OutputEndpoints.predicted_text will
            utf8 encoded strings corresponding to the character ids returned by
            OutputEndpoints.predicted_chars (by default the predicted_text contains
            an empty vector).
            NOTE: Make sure you call tf.tables_initializer().run() if the charset
            specified.
        """
        super(Model, self).__init__()
        self._params = ModelParams(
                num_char_classes = num_char_classes,
                seq_length = seq_length,
                num_views = num_views,
                null_code = null_code)
        self._mparams = self.default_mparams()
        if mparams:
            self._mparams.update(mparams)
        self._charset = charset


    def default_mparams(self):
        return {
            'conv_tower_fn'        :
                ConvTowerParams(final_endpoint = 'Mixed_5d'),
            'sequence_logit_fn'    :
                SequenceLogitsParams(
                        use_attention = True,
                        use_autoregression = True,
                        num_lstm_units = 256,
                        weight_decay = 0.00004,
                        lstm_state_clip_value = 10.0),
            'sequence_loss_fn'     :
                SequenceLossParams(
                        label_smoothing = 0.1,
                        ignore_nulls = True,
                        average_across_timesteps = False),
            'encode_coordinates_fn': EncodeCoordinatesParams(enabled = True)
        }


    def set_mparam(self, function, **kwargs):
        self._mparams[function] = self._mparams[function]._replace(**kwargs)


    def conv_tower_fn(self, images, is_training = True, reuse = None):
        """Computes convolutional features using the InceptionV3 model.

        Args:
          images: A tensor of shape [batch_size, height, width, channels].
          is_training: whether is training or not.
          reuse: whether or not the network and its variables should be reused. To
            be able to reuse 'scope' must be given.

        Returns:
          A tensor of shape [batch_size, OH, OW, N], where OWxOH is resolution of
          output feature map and N is number of output features (depends on the
          network architecture).
        """
        mparams = self._mparams['conv_tower_fn']
        logging.debug('Using final_endpoint=%s', mparams.final_endpoint)
        with tf.variable_scope('conv_tower_fn/INCE'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            with slim.arg_scope(inception.inception_v3_arg_scope()):
                with slim.arg_scope([slim.batch_norm, slim.dropout],
                                    is_training = is_training):
                    net, _ = inception.inception_v3_base(
                            images, final_endpoint = mparams.final_endpoint)
            return net


    def _create_lstm_inputs(self, net):
        """Splits an input tensor into a list of tensors (features).

        Args:
          net: A feature map of shape [batch_size, num_features, feature_size].

        Raises:
          AssertionError: if num_features is less than seq_length.

        Returns:
          A list with seq_length tensors of shape [batch_size, feature_size]
        """
        num_features = net.get_shape().dims[1].value
        if num_features < self._params.seq_length:
            raise AssertionError('Incorrect dimension #1 of input tensor'
                                 ' %d should be bigger than %d (shape=%s)' %
                                 (num_features, self._params.seq_length,
                                  net.get_shape()))
        elif num_features > self._params.seq_length:
            logging.warning('Ignoring some features: use %d of %d (shape=%s)',
                            self._params.seq_length, num_features, net.get_shape())
            net = tf.slice(net, [0, 0, 0], [-1, self._params.seq_length, -1])

        return tf.unstack(net, axis = 1)


    def sequence_logit_fn(self, net, labels):
        mparams = self._mparams['sequence_logit_fn']
        with tf.variable_scope('sequence_logit_fn/SQLR'):
            layer = sequence_layers.SequenceLayer(net, labels, self._params, mparams)
            return layer.create_logits()


    def max_pool_views(self, nets_list):
        """Max pool across all nets in spatial dimensions.

        Args:
          nets_list: A list of 4D tensors with identical size.

        Returns:
          A tensor with the same size as any input tensors.
        """
        batch_size, height, width, num_features = [
            d.value for d in nets_list[0].get_shape().dims
        ]
        xy_flat_shape = (batch_size, 1, height * width, num_features)
        nets_for_merge = []
        with tf.variable_scope('max_pool_views', values = nets_list):
            for net in nets_list:
                nets_for_merge.append(tf.reshape(net, xy_flat_shape))
            merged_net = tf.concat(nets_for_merge, 1)
            net = slim.max_pool2d(
                    merged_net, kernel_size = [len(nets_list), 1], stride = 1)
            net = tf.reshape(net, (batch_size, height, width, num_features))
        return net


    def pool_views_fn(self, nets):
        """Combines output of multiple convolutional towers into a single tensor.

        It stacks towers one on top another (in height dim) in a 4x1 grid.
        The order is arbitrary design choice and shouldn't matter much.

        Args:
          nets: list of tensors of shape=[batch_size, height, width, num_features].

        Returns:
          A tensor of shape [batch_size, seq_length, features_size].
        """
        with tf.variable_scope('pool_views_fn/STCK'):
            net = tf.concat(nets, 1)
            batch_size = net.get_shape().dims[0].value
            feature_size = net.get_shape().dims[3].value
            return tf.reshape(net, [batch_size, -1, feature_size])


    def encode_coordinates_fn(self, net):
        """Adds one-hot encoding of coordinates to different views in the networks.

        For each "pixel" of a feature map it adds a onehot encoded x and y
        coordinates.

        Args:
          net: a tensor of shape=[batch_size, height, width, num_features]

        Returns:
          a tensor with the same height and width, but altered feature_size.
        """
        mparams = self._mparams['encode_coordinates_fn']
        if mparams.enabled:
            batch_size, h, w, _ = net.shape.as_list()
            x, y = tf.meshgrid(tf.range(w), tf.range(h))
            w_loc = slim.one_hot_encoding(x, num_classes = w)
            h_loc = slim.one_hot_encoding(y, num_classes = h)
            loc = tf.concat([h_loc, w_loc], 2)
            loc = tf.tile(tf.expand_dims(loc, 0), [batch_size, 1, 1, 1])
            return tf.concat([net, loc], 3)
        else:
            return net


    def create_base(self, images, labels, scope = 'AttentionOcr_v1', reuse = None):
        """Creates a base part of the Model (no gradients, losses or summaries).

        Args:
          images: A tensor of shape [batch_size, height, width, channels].
          labels: Ground truth labels.
          scope: Optional variable_scope.
          reuse: whether or not the network and its variables should be reused. To
            be able to reuse 'scope' must be given.

        Returns:
          A named tuple OutputEndpoints.
        """
        is_training = labels is not None
        with tf.variable_scope(scope, reuse = reuse):
            net = self.conv_tower_fn(images, is_training, False)
            net = self.encode_coordinates_fn(net)
            net = self.pool_views_fn(net)
            logit = self.sequence_logit_fn(net, labels)
            prob = tf.reduce_max(input_tensor = logit, axis = 1)
        return prob


    def create_loss(self, predicted, ground_truth):
        """Creates all losses required to train the model.

        Args:
          endpoints: Model namedtuple.

        Returns:
          Total loss.
        """

        loss = -tf.reduce_sum(ground_truth * tf.log(predicted) + (1 - ground_truth) * tf.log(1 - predicted))
        positive_acc = tf.reduce_sum(ground_truth * predicted) / tf.reduce_sum(ground_truth)
        negative_acc = tf.reduce_sum((1 - ground_truth) * (1 - predicted)) / tf.reduce_sum(1 - ground_truth)

        tf.summary.scalar('Loss', loss)
        with tf.name_scope('Accuracy'):
            tf.summary.scalar('positive_accuracy', positive_acc)
            tf.summary.scalar('negative_accuracy', negative_acc)
            tf.summary.scalar('total_accuracy', (positive_acc + negative_acc) / 2)

        return loss


    def create_init_fn_to_restore(self, master_checkpoint, inception_checkpoint = None):
        """Creates an init operations to restore weights from various checkpoints.

        Args:
          master_checkpoint: path to a checkpoint which contains all weights for
            the whole model.
          inception_checkpoint: path to a checkpoint which contains weights for the
            inception part only.

        Returns:
          a function to run initialization ops.
        """
        all_assign_ops = []
        all_feed_dict = {}

        def assign_from_checkpoint(variables, checkpoint):
            logging.info('Request to re-store %d weights from %s', len(variables), checkpoint)
            if not variables:
                logging.error('Can\'t find any variables to restore.')
                sys.exit(1)
            assign_op, feed_dict = slim.assign_from_checkpoint(checkpoint, variables)
            all_assign_ops.append(assign_op)
            all_feed_dict.update(feed_dict)

        if master_checkpoint:
            assign_from_checkpoint(utils.variables_to_restore(), master_checkpoint)

        if inception_checkpoint:
            variables = utils.variables_to_restore('AttentionOcr_v1/conv_tower_fn/INCE', strip_scope = True)
            assign_from_checkpoint(variables, inception_checkpoint)

        def init_assign_fn(sess):
            logging.info('Restoring checkpoint(s)')
            sess.run(all_assign_ops, all_feed_dict)

        return init_assign_fn
