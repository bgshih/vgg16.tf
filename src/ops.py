import tensorflow as tf
import math


def conv2d(x, n_in, n_out, k, s, p='SAME', bias=True, scope=None):
  with tf.variable_scope(scope or 'Conv2D'):
    kernel_init_std = math.sqrt(2.0 / (k * k * n_in))
    kernel = tf.get_variable('Weight', shape=[k,k,n_in,n_out],
      initializer=tf.truncated_normal_initializer(0.0, kernel_init_std))
    tf.add_to_collection('Weights', kernel)
    y = tf.nn.conv2d(x, kernel, [1,1,s,s], padding=p, data_format='NCHW')
    if bias == True:
      bias = tf.get_variable('Bias', shape=[n_out],
        initializer=tf.constant_initializer(0.0))
      tf.add_to_collection('Biases', bias)
      y = tf.nn.bias_add(y, bias, data_format='NCHW')
  return y

def linear(x, n_in, n_out, bias=True, scope=None):
  with tf.variable_scope(scope or 'Linear'):
    weight_init_std = math.sqrt(1.0 / n_out)
    weight = tf.get_variable('Weight', shape=[n_in,n_out],
      initializer=tf.truncated_normal_initializer(0.0, weight_init_std))
    tf.add_to_collection('Weights', weight)
    y = tf.matmul(x, weight)
    if bias == True:
      bias = tf.get_variable('Bias', shape=[n_out],
        initializer=tf.constant_initializer(0.0))
      tf.add_to_collection('Biases', bias)
      y = y + bias
  return y

def mlp(x, n_in, n_hidden, n_out, activation=tf.nn.relu, scope=None):
  with tf.variable_scope(scope or 'Mlp'):
    y = linear(x, n_in, n_hidden, scope='Linear1')
    y = activation(y)
    y = linear(y, n_hidden, n_out, scope='Linear2')
  return y
