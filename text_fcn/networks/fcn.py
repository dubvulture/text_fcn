from __future__ import absolute_import
from six.moves import range

import tensorflow as tf

from text_fcn import tf_utils
from text_fcn.networks.vgg_net import create_vgg_net


def create_fcn(placeholder, keep_prob, classes):
    """
    Setup the main conv/deconv network
    """
    with tf.variable_scope('FCN8'):
        vgg_net = create_vgg_net(placeholder)
        conv_final = vgg_net['conv5_4']
        output = tf_utils.max_pool_2x2(conv_final)

        conv_shapes = [
            [7, 7, 512, 4096],
            [1, 1, 4096, 4096]
        ]

        for i, conv_shape in enumerate(conv_shapes):
            with tf.variable_scope('conv%d' % (i + 6)) as name:
                W = tf_utils.weight_variable(conv_shape)
                b = tf_utils.bias_variable(conv_shape[-1:])
                output = tf_utils.conv2d_basic(output, W, b)
            with tf.variable_scope('relu%d' % (i + 6)):
                output = tf.nn.relu(output)
            tf_utils.add_activation_summary(output, collections=['train'])
            output = tf.nn.dropout(output, keep_prob=keep_prob)

        pool4 = vgg_net['pool4']
        pool3 = vgg_net['pool3']

        W_shapes = [
            [4, 4, pool4.get_shape()[3].value, classes],
            [4, 4, pool3.get_shape()[3].value, pool4.get_shape()[3].value],
        ]
        deconv_shapes = [tf.shape(pool4), tf.shape(pool3)]

        out_channels = [classes, 2]
        branch = [None, None]
        pred = [None, None]

        for i, t in enumerate(['semantic', 'instance']):
            with tf.variable_scope('conv8_%s' % t):
                W = tf_utils.weight_variable([1, 1, 4096, out_channels[i]])
                b = tf_utils.bias_variable([out_channels[i]])
                branch[i] = tf_utils.conv2d_basic(output, W, b)

            for j in range(2):
                with tf.variable_scope('deconv%d_%s' % (j + 1, t)):
                    W = tf_utils.weight_variable(W_shapes[j])
                    b = tf_utils.bias_variable([W_shapes[j][2]])
                    branch[i] = tf_utils.conv2d_transpose_strided(
                        branch[i], W, b,
                        output_shape=deconv_shapes[j], stride=2)
                with tf.variable_scope('skip%d_%s' % (j + 1, t)):
                    branch[i] = tf.add(branch[i], vgg_net['pool%d' % (4 - j)])

            with tf.variable_scope('deconv3_%s' % t):
                W = tf_utils.weight_variable([16, 16, out_channels[i], pool3.get_shape()[3].value])
                b = tf_utils.bias_variable([out_channels[i]])
                branch[i] = tf_utils.conv2d_transpose_strided(
                        branch[i], W, b,
                        output_shape=tf.stack([
                            tf.shape(placeholder)[0], tf.shape(placeholder)[1],
                            tf.shape(placeholder)[2], out_channels[i]]),
                        stride=8)

    return branch
