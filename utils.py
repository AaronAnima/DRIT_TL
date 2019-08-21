# from weightLightModels.wrapper import Wrapper
from tensorlayer.layers import Layer
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.ops import nn_impl
from tensorflow.python.keras import initializers

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import MeanPool2d, ExpandDims, Tile, UpSampling2d, Elementwise, \
    GlobalMeanPool2d, BatchNorm2d, Lambda, Input, Dense, DeConv2d, Reshape,\
    Conv2d, Flatten, Concat, GaussianNoise, LayerNorm
from tensorlayer.layers import (SubpixelConv2d, ExpandDims)
from tensorlayer.layers import DeConv2d
from config import flags
from tensorlayer.models import Model
import os

# https://github.com/tensorlayer/tensorlayer/pull/1013/files#diff-d7edcbe6c6016196e9a72ea65f851520R37

w_init = tf.random_normal_initializer(stddev=0.02)
g_init = tf.random_normal_initializer(1., 0.02)
lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)  # tl.act.lrelu(x, 0.2)


def count_weights(model):
    n_weights = 0
    for i, w in enumerate(model.all_weights):
        n = 1
        # for s in p.eval().shape:
        for s in w.get_shape():
            try:
                s = int(s)
            except:
                s = 1
            if s:
                n = n * s
        n_weights = n_weights + n
    print("num of weights (parameters) %d" % n_weights)
    return n_weights


def spectral_norm(w, u, iteration=1):  # https://github.com/taki0112/Spectral_Normalization-Tensorflow/blob/master/spectral_norm.py
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    # u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)
    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm


class SpectralNormConv2d(Conv2d):
    """
    The :class:`SpectralNormConv2d` class is a Conv2d layer for with Spectral Normalization.
    ` Spectral Normalization for Generative Adversarial Networks (ICLR 2018) <https://openreview.net/forum?id=B1QRgziT-&noteId=BkxnM1TrM>`__

    Parameters
    ----------
    n_filter : int
        The number of filters.
    filter_size : tuple of int
        The filter size (height, width).
    strides : tuple of int
        The sliding window strides of corresponding input dimensions.
        It must be in the same order as the ``shape`` parameter.
    dilation_rate : tuple of int
        Specifying the dilation rate to use for dilated convolution.
    act : activation function
        The activation function of this layer.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        "channels_last" (NHWC, default) or "channels_first" (NCHW).
    W_init : initializer
        The initializer for the the weight matrix.
    b_init : initializer or None
        The initializer for the the bias vector. If None, skip biases.
    in_channels : int
        The number of in channels.
    name : None or str
        A unique layer name.
    """
    def __init__(
            self,
            n_filter=32,
            filter_size=(3, 3),
            strides=(1, 1),
            act=None,
            padding='SAME',
            data_format='channels_last',
            dilation_rate=(1, 1),
            W_init=tl.initializers.truncated_normal(stddev=0.02),
            b_init=tl.initializers.constant(value=0.0),
            in_channels=None,
            name=None
    ):
        super(SpectralNormConv2d, self).__init__(n_filter=n_filter, filter_size=filter_size,
            strides=strides, act=act, padding=padding, data_format=data_format,
            dilation_rate=dilation_rate, W_init=W_init, b_init=b_init, in_channels=in_channels,
            name=name)
        # logging.info(
        #     "    It is a SpectralNormConv2d %s: n_filter: %d filter_size: %s strides: %s pad: %s act: %s" % (
        #         self.name, n_filter, str(filter_size), str(strides), padding,
        #         self.act.__name__ if self.act is not None else 'No Activation'
        #     )
        # )
        if self.in_channels:
            self.build(None)
            self._built = True

    def build(self, inputs_shape): # # override
        self.u =  self._get_weights("u", shape=[1, self.n_filter], init=tf.random_normal_initializer(), trainable=False) # tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
        # self.s =  self._get_weights("sigma", shape=[1, ], init=tf.random_normal_initializer(), trainable=False)
        super(SpectralNormConv2d, self).build(inputs_shape)

    def forward(self, inputs): # override
        self.W_norm = spectral_norm(self.W, self.u)
        # self.W_norm = spectral_norm(self.W, self.u, self.s)
        # return super(SpectralNormConv2d, self).forward(inputs)
        outputs = tf.nn.conv2d(
            input=inputs,
            filters=self.W_norm,
            strides=self._strides,
            padding=self.padding,
            data_format=self.data_format,  #'NHWC',
            dilations=self._dilation_rate,  #[1, 1, 1, 1],
            name=self.name,
        )
        if self.b_init:
            outputs = tf.nn.bias_add(outputs, self.b, data_format=self.data_format, name='bias_add')
        if self.act:
            outputs = self.act(outputs)
        return outputs
