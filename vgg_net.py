import numpy as np
import os
import scipy
import tensorflow as tf

import TensorflowUtils as utils
from coco_utils import maybe_download_and_extract



class VGG_NET(object):
    """
    Object representing a partial VGG19-Net for our purposes
    """

    MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/' \
                'imagenet-vgg-verydeep-19.mat'
    
    LAYERS = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )


    def __init__(self, placeholder, model_dir='Model_zoo/'):
        """
        :param placeholder: tf.placeholder where we will operate
        :param model_dir:   directory where to find the model .mat file
        """
        self.model_dir = model_dir
        weights, mean_pixel = self._get_model_attr()
        self.net = self._setup_net(placeholder, weights)

    def _setup_net(self, placeholder, weights):
        """
        Returns the network built with given weights
        """
        net = {}
        for i, name in enumerate(VGG_NET.LAYERS):
            kind = name[:4]
            if kind == 'conv':
                kernels, bias = weights[i][0][0][0][0]
                # matconvnet: [width, height, in_channels, out_channels]
                # tensorflow: [height, width, in_channels, out_channels]
                kernels = utils.get_variable(
                    np.transpose(kernels, (1, 0, 2, 3)),
                    name=name + "_w")
                bias = utils.get_variable(
                    bias.reshape(-1),
                    name=name + "_b")
                placeholder = utils.conv2d_basic(placeholder, kernels, bias)
            elif kind == 'relu':
                placeholder = tf.nn.relu(placeholder, name=name)
                utils.add_activation_summary(placeholder, collections=['train'])
            elif kind == 'pool':
                # VGG specifies max_pool... so why avg? Let's wait for response
                #placeholder = utils.max_pool_2x2(placeholder)
                placeholder = utils.avg_pool_2x2(placeholder)
            net[name] = placeholder

        return net



    def _get_model_attr(self):
        """
        Get the model file if needed and return VGG19 weights and mean pixel
        """
        maybe_download_and_extract(self.model_dir, VGG_NET.MODEL_URL)
        filename = VGG_NET.MODEL_URL.split("/")[-1]
        filepath = os.path.join(self.model_dir, filename)
        if not os.path.exists(filepath):
            raise IOError("VGG Model not found!")
        
        data = scipy.io.loadmat(filepath)
        mean = data['normalization']['averageImage'][0,0]
        mean_pixel = np.mean(mean, axis=(0, 1))
        weights = np.squeeze(data['layers'])

        return weights, mean_pixel

