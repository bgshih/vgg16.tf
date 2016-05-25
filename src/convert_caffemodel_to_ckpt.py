import sys, os
import tensorflow as tf
import joblib
import numpy as np
import argparse

import model_vgg16
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_scope', default='Vgg16',
                    help='Scope for the tensorflow model.')
parser.add_argument('--ckpt_path', default='/tmp/VGG_ILSVRC_16_layers.ckpt',
                    help='Checkpoint save path.')
parser.add_argument('--caffe_weights_path', default='/tmp/VGG_ILSVRC_16_layers_weights.pkl',
                    help='weights dump path.')
args = parser.parse_args()


def convert_caffemodel_to_ckpt():
  caffe_weights = joblib.load(args.caffe_weights_path)

  # create network
  vgg16 = model_vgg16.Vgg16Model()
  model_scope = args.model_scope
  vgg16(tf.placeholder(tf.float32), scope=model_scope)

  # auxillary functions for conversion
  def load_conv_weight(target_name, src_name):
    target_name = model_scope + '/' + target_name
    src = np.transpose(caffe_weights[src_name][0], (2,3,1,0))
    return tf.assign(tf.get_variable(target_name), src)
  def load_conv_bias(target_name, src_name):
    target_name = model_scope + '/' + target_name
    src = caffe_weights[src_name][1]
    return tf.assign(tf.get_variable(target_name), src)
  def load_fc_weight(target_name, src_name):
    target_name = model_scope + '/' + target_name
    src = np.transpose(caffe_weights[src_name][0], (1,0))
    return tf.assign(tf.get_variable(target_name), src)
  def load_fc_bias(target_name, src_name):
    target_name = model_scope + '/' + target_name
    src = caffe_weights[src_name][1]
    return tf.assign(tf.get_variable(target_name), src)

  # loding caffemodel weights
  with tf.Session() as session:
    tf.get_variable_scope().reuse_variables()
    assigns = [
      load_conv_weight('conv1_1/Conv2D/Weight', 'conv1_1'),
      load_conv_bias('conv1_1/Conv2D/Bias', 'conv1_1'),
      load_conv_weight('conv1_2/Conv2D/Weight', 'conv1_2'),
      load_conv_bias('conv1_2/Conv2D/Bias', 'conv1_2'),
      load_conv_weight('conv2_1/Conv2D/Weight', 'conv2_1'),
      load_conv_bias('conv2_1/Conv2D/Bias', 'conv2_1'),
      load_conv_weight('conv2_2/Conv2D/Weight', 'conv2_2'),
      load_conv_bias('conv2_2/Conv2D/Bias', 'conv2_2'),
      load_conv_weight('conv3_1/Conv2D/Weight', 'conv3_1'),
      load_conv_bias('conv3_1/Conv2D/Bias', 'conv3_1'),
      load_conv_weight('conv3_2/Conv2D/Weight', 'conv3_2'),
      load_conv_bias('conv3_2/Conv2D/Bias', 'conv3_2'),
      load_conv_weight('conv3_3/Conv2D/Weight', 'conv3_3'),
      load_conv_bias('conv3_3/Conv2D/Bias', 'conv3_3'),
      load_conv_weight('conv4_1/Conv2D/Weight', 'conv4_1'),
      load_conv_bias('conv4_1/Conv2D/Bias', 'conv4_1'),
      load_conv_weight('conv4_2/Conv2D/Weight', 'conv4_2'),
      load_conv_bias('conv4_2/Conv2D/Bias', 'conv4_2'),
      load_conv_weight('conv4_3/Conv2D/Weight', 'conv4_3'),
      load_conv_bias('conv4_3/Conv2D/Bias', 'conv4_3'),
      load_conv_weight('conv5_1/Conv2D/Weight', 'conv5_1'),
      load_conv_bias('conv5_1/Conv2D/Bias', 'conv5_1'),
      load_conv_weight('conv5_2/Conv2D/Weight', 'conv5_2'),
      load_conv_bias('conv5_2/Conv2D/Bias', 'conv5_2'),
      load_conv_weight('conv5_3/Conv2D/Weight', 'conv5_3'),
      load_conv_bias('conv5_3/Conv2D/Bias', 'conv5_3'),
      load_fc_weight('fc6/Linear/Weight', 'fc6'),
      load_fc_bias('fc6/Linear/Bias', 'fc6'),
      load_fc_weight('fc7/Linear/Weight', 'fc7'),
      load_fc_bias('fc7/Linear/Bias', 'fc7'),
      load_fc_weight('fc8/Linear/Weight', 'fc8'),
      load_fc_bias('fc8/Linear/Bias', 'fc8'),
    ]
    with tf.control_dependencies(assigns):
      load_op = tf.no_op(name='LoadOp')
    session.run(load_op)

    # save checkpoint
    saver = tf.train.Saver()
    saver.save(session, args.ckpt_path)


if __name__ == '__main__':
  convert_caffemodel_to_ckpt()
