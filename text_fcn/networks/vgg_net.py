from __future__ import absolute_import

import os

import numpy as np
import scipy.io
import tensorflow as tf

from text_fcn import coco_utils
from text_fcn import tf_utils


MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/' \
            'imagenet-vgg-verydeep-16.mat'

LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3'
)


def create_vgg_net(placeholder, model_dir='text_fcn/networks/Model_zoo/'):
    """
    :param placeholder: tf.placeholder where we will operate
    :param model_dir:   directory where to find the model .mat file
    """
    return _setup_net(placeholder, *_get_model_attr(model_dir))


def _setup_net(placeholder, weights, mean_pixel):
    """
    Returns the cnn built with given weights and normalized with mean_pixel
    """
    net = {}
    placeholder -= mean_pixel
    global LAYERS
    for i, name in enumerate(LAYERS):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: [width, height, in_channels, out_channels]
            # tensorflow: [height, width, in_channels, out_channels]
            kernels = tf_utils.get_variable(
                np.transpose(kernels, (1, 0, 2, 3)),
                name=name + "_w")
            bias = tf_utils.get_variable(
                bias.reshape(-1),
                name=name + "_b")
            placeholder = tf_utils.conv2d_basic(placeholder, kernels, bias)
        elif kind == 'relu':
            placeholder = tf.nn.relu(placeholder, name=name)
            tf_utils.add_activation_summary(placeholder, collections=['train'])
        elif kind == 'pool':
            # VGG specifies max_pool... so why avg? Let's wait for response
            placeholder = tf_utils.max_pool_2x2(placeholder)
            # placeholder = tf_utils.avg_pool_2x2(placeholder)
        net[name] = placeholder

    return net


def _get_model_attr(model_dir):
    """
    Get the model file if needed and return VGG19 weights and mean pixel
    """
    global MODEL_URL
    coco_utils.maybe_download_and_extract(model_dir, MODEL_URL)
    filename = MODEL_URL.split("/")[-1]
    filepath = os.path.join(model_dir, filename)
    if not os.path.exists(filepath):
        raise IOError("VGG Model not found.")
    
    data = scipy.io.loadmat(filepath)
    mean = data['normalization']['averageImage'][0,0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = np.squeeze(data['layers'])

    return weights, mean_pixel

