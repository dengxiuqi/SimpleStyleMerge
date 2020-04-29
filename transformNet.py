import tensorflow as tf
import numpy as np
import scipy.io
import scipy, imageio
import os, sys


def transform_net(img, relu=True, inorm=True):
    conv1 = conv_layer(img, 9, 32, 1, relu=relu, inorm=inorm)
    conv2 = conv_layer(conv1, 3, 64, 2, relu=relu, inorm=inorm)
    conv3 = conv_layer(conv2, 3, 128, 2, relu=relu, inorm=inorm)
    resid1 = residual_block(conv3, relu=relu, inorm=inorm)
    resid2 = residual_block(resid1, relu=relu, inorm=inorm)
    resid3 = residual_block(resid2, relu=relu, inorm=inorm)
    resid4 = residual_block(resid3, relu=relu, inorm=inorm)
    resid5 = residual_block(resid4, relu=relu, inorm=inorm)
    conv_t1 = conv_tranpose_layer(resid5, 3, 64, 2, relu=relu, inorm=inorm)
    conv_t2 = conv_tranpose_layer(conv_t1, 3, 32, 2, relu=relu, inorm=inorm)
    conv_t3 = conv_tranpose_layer(conv_t2, 9, 3, 1, relu=False, inorm=inorm)
    output = tf.nn.tanh(conv_t3)
    return output


def conv_layer(input, filter_size, filter_num, stride_size, relu=True, inorm=True):
    kernel_size = [filter_size, filter_size, input.shape[-1].value, filter_num]
    kernel = tf.Variable(tf.truncated_normal(kernel_size, stddev=0.05))
    bias = tf.Variable(tf.constant(0.05, shape=[filter_num]))
    conv = tf.nn.conv2d(input, kernel, strides=[1, stride_size, stride_size, 1], padding='SAME')
    output = tf.nn.bias_add(conv, bias)
    if inorm:
        output = instance_norm(output)
    if relu:
        output = tf.nn.relu(output)
    return output


def residual_block(input, filter_size=3, stride_size=1, relu=True, inorm=True):
    conv = conv_layer(input, filter_size, input.shape[-1].value, stride_size, relu=relu, inorm=inorm)
    output = input + conv_layer(conv, filter_size, input.shape[-1].value, stride_size, relu=False, inorm=inorm)
    return output


def conv_tranpose_layer(input, filter_size, filter_num, stride_size, relu=True, inorm=True):
    batch, rows, cols, in_channels = [i.value for i in input.shape]

    kernel_size = [filter_size, filter_size, filter_num, in_channels]
    kernel = tf.Variable(tf.truncated_normal(kernel_size, stddev=0.05))
    bias = tf.Variable(tf.constant(0.05, shape=[filter_num]))

    output_shape = [batch, rows * stride_size, cols * stride_size, filter_num]
    conv = tf.nn.conv2d_transpose(input, kernel, output_shape,
                                  strides=[1, stride_size, stride_size, 1], padding='SAME')
    output = tf.nn.bias_add(conv, bias)
    if inorm:
        output = instance_norm(output)
    if relu:
        output = tf.nn.relu(output)
    return output


def instance_norm(net, train=True):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift
