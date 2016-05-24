import sys, os
import tensorflow as tf
import joblib
import numpy as np

import ops


class Vgg16Model():
  def __init__(self):
    self.image_mean = np.array([103.939, 116.779, 123.68])

  def _vgg_conv_relu(self, x, n_in, n_out, scope):
    with tf.variable_scope(scope):
      conv = ops.conv2d(x, n_in, n_out, 3, 1, p='SAME')
      relu = tf.nn.relu(conv)
    return relu

  def _vgg_max_pool(self, x, scope):
    with tf.variable_scope(scope):
      pool = tf.nn.max_pool(x, [1,1,2,2], [1,1,2,2],
        padding='SAME', data_format='NCHW')
    return pool

  def _vgg_fully_connected(self, x, n_in, n_out, scope):
    with tf.variable_scope(scope):
      fc = ops.linear(x, n_in, n_out)
    return fc

  def __call__(self, x, scope=None):
    with tf.variable_scope(scope or 'Vgg16'):
      # conv stage 1
      relu1_1 = self._vgg_conv_relu(x, 3, 64, 'conv1_1')
      relu1_2 = self._vgg_conv_relu(relu1_1, 64, 64, 'conv1_2')
      pool1 = self._vgg_max_pool(relu1_2, 'pool1')
      # conv stage 2
      relu2_1 = self._vgg_conv_relu(pool1, 64, 128, 'conv2_1')
      relu2_2 = self._vgg_conv_relu(relu2_1, 128, 128, 'conv2_2')
      pool2 = self._vgg_max_pool(relu2_2, 'pool2')
      # conv stage 3
      relu3_1 = self._vgg_conv_relu(pool2, 128, 256, 'conv3_1')
      relu3_2 = self._vgg_conv_relu(relu3_1, 256, 256, 'conv3_2')
      relu3_3 = self._vgg_conv_relu(relu3_2, 256, 256, 'conv3_3')
      pool3 = self._vgg_max_pool(relu3_3, 'pool3')
      # conv stage 4
      relu4_1 = self._vgg_conv_relu(pool3, 256, 512, 'conv4_1')
      relu4_2 = self._vgg_conv_relu(relu4_1, 512, 512, 'conv4_2')
      relu4_3 = self._vgg_conv_relu(relu4_2, 512, 512, 'conv4_3')
      pool4 = self._vgg_max_pool(relu4_3, 'pool4')
      # conv stage 5
      relu5_1 = self._vgg_conv_relu(pool4, 512, 512, 'conv5_1')
      relu5_2 = self._vgg_conv_relu(relu5_1, 512, 512, 'conv5_2')
      relu5_3 = self._vgg_conv_relu(relu5_2, 512, 512, 'conv5_3')
      pool5 = self._vgg_max_pool(relu5_3, 'pool4')
      # fc6
      n_conv_out = 7*7*512
      flatten = tf.reshape(pool5, [-1,n_conv_out])
      fc6 = self._vgg_fully_connected(flatten, n_conv_out, 4096, scope='fc6')
      relu_6 = tf.nn.relu(fc6)
      # fc7
      fc7 = self._vgg_fully_connected(fc6, 4096, 4096, scope='fc7')
      relu_7 = tf.nn.relu(fc7)
      # fc8, prob
      fc8 = self._vgg_fully_connected(relu_7, 4096, 1000, scope='fc8')
      prob = tf.nn.softmax(fc8)
    return prob

  def load_weights_from_caffemodel(self, session):
    caffe_weights = joblib.load('../data/VGG_ILSVRC_16_layers.pkl')

    def _load_conv_weight(target_name, src_name):
      src = np.transpose(caffe_weights[src_name][0], (2,3,1,0))
      return tf.assign(tf.get_variable(target_name), src)
    def _load_conv_bias(target_name, src_name):
      src = caffe_weights[src_name][1]
      return tf.assign(tf.get_variable(target_name), src)
    def _load_fc_weight(target_name, src_name):
      src = np.transpose(caffe_weights[src_name][0], (1,0))
      return tf.assign(tf.get_variable(target_name), src)
    def _load_fc_bias(target_name, src_name):
      src = caffe_weights[src_name][1]
      return tf.assign(tf.get_variable(target_name), src)

    tf.get_variable_scope().reuse_variables()
    assigns = [
      _load_conv_weight('Vgg16/conv1_1/Conv2D/Weight', 'conv1_1'),
      _load_conv_bias('Vgg16/conv1_1/Conv2D/Bias', 'conv1_1'),
      _load_conv_weight('Vgg16/conv1_2/Conv2D/Weight', 'conv1_2'),
      _load_conv_bias('Vgg16/conv1_2/Conv2D/Bias', 'conv1_2'),
      _load_conv_weight('Vgg16/conv2_1/Conv2D/Weight', 'conv2_1'),
      _load_conv_bias('Vgg16/conv2_1/Conv2D/Bias', 'conv2_1'),
      _load_conv_weight('Vgg16/conv2_2/Conv2D/Weight', 'conv2_2'),
      _load_conv_bias('Vgg16/conv2_2/Conv2D/Bias', 'conv2_2'),
      _load_conv_weight('Vgg16/conv3_1/Conv2D/Weight', 'conv3_1'),
      _load_conv_bias('Vgg16/conv3_1/Conv2D/Bias', 'conv3_1'),
      _load_conv_weight('Vgg16/conv3_2/Conv2D/Weight', 'conv3_2'),
      _load_conv_bias('Vgg16/conv3_2/Conv2D/Bias', 'conv3_2'),
      _load_conv_weight('Vgg16/conv3_3/Conv2D/Weight', 'conv3_3'),
      _load_conv_bias('Vgg16/conv3_3/Conv2D/Bias', 'conv3_3'),
      _load_conv_weight('Vgg16/conv4_1/Conv2D/Weight', 'conv4_1'),
      _load_conv_bias('Vgg16/conv4_1/Conv2D/Bias', 'conv4_1'),
      _load_conv_weight('Vgg16/conv4_2/Conv2D/Weight', 'conv4_2'),
      _load_conv_bias('Vgg16/conv4_2/Conv2D/Bias', 'conv4_2'),
      _load_conv_weight('Vgg16/conv4_3/Conv2D/Weight', 'conv4_3'),
      _load_conv_bias('Vgg16/conv4_3/Conv2D/Bias', 'conv4_3'),
      _load_conv_weight('Vgg16/conv5_1/Conv2D/Weight', 'conv5_1'),
      _load_conv_bias('Vgg16/conv5_1/Conv2D/Bias', 'conv5_1'),
      _load_conv_weight('Vgg16/conv5_2/Conv2D/Weight', 'conv5_2'),
      _load_conv_bias('Vgg16/conv5_2/Conv2D/Bias', 'conv5_2'),
      _load_conv_weight('Vgg16/conv5_3/Conv2D/Weight', 'conv5_3'),
      _load_conv_bias('Vgg16/conv5_3/Conv2D/Bias', 'conv5_3'),
      _load_fc_weight('Vgg16/fc6/Linear/Weight', 'fc6'),
      _load_fc_bias('Vgg16/fc6/Linear/Bias', 'fc6'),
      _load_fc_weight('Vgg16/fc7/Linear/Weight', 'fc7'),
      _load_fc_bias('Vgg16/fc7/Linear/Bias', 'fc7'),
      _load_fc_weight('Vgg16/fc8/Linear/Weight', 'fc8'),
      _load_fc_bias('Vgg16/fc8/Linear/Bias', 'fc8'),
    ]
    with tf.control_dependencies(assigns):
      load_op = tf.no_op(name='LoadOp')
    session.run(load_op)

  def image_preprococess_testing(self, image):
    """
    # 
    ARGS:
      image: 224x224 BGR image with HWC layout
    RETURN:
      image: 224x224 normalized BGR image with CHW layout
    """
    image_mean = tf.reshape(tf.constant(self.image_mean, dtype=tf.float32), [1,1,3])
    image = image - image_mean
    image = tf.transpose(image, [2,0,1])
    return image
