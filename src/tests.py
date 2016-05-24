import math
import tensorflow as tf
import numpy as np
import skimage
import skimage.io, skimage.transform

import model_vgg16


def test_forward():
  vgg16 = model_vgg16.Vgg16Model()

  images = tf.random_normal(shape=[128,3,224,224])
  prob = vgg16(images)
  with tf.Session() as session:
    session.run(tf.initialize_all_variables())

    print('Loading model from caffe...')
    vgg16.load_weights_from_caffemodel(session)
    print('Model loaded...')

    session_outputs = session.run(prob)
    print(session_outputs.shape)


def test_classify_image():

  def load_image_and_preprocess(fname):
    image = skimage.io.imread(fname)
    assert(image.ndim == 3)
    # scale image so that the shorter side is 224
    image_h, image_w = image.shape[0], image.shape[1]
    shorter_side = min(image_h, image_w)
    scale = 224.0 / shorter_side
    image = skimage.transform.rescale(image, scale)
    image_h, image_w = image.shape[0], image.shape[1]
    # center crop
    crop_x = (image_w - 224) / 2
    crop_y = (image_h - 224) / 2
    image = image[crop_y:crop_y+224,crop_x:crop_x+224,:]
    # RGB -> BGR
    image = image[:,:,::-1]
    image *= 255.0
    return image

  image = load_image_and_preprocess('cat.png')

  vgg16 = model_vgg16.Vgg16Model()

  input_image = tf.placeholder(tf.float32, shape=[224,224,3])
  test_image = vgg16.image_preprococess_testing(input_image)
  image_batch = tf.reshape(test_image, [1,3,224,224])
  prob = vgg16(image_batch)

  with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    vgg16.load_weights_from_caffemodel(session)

    session_outputs = session.run([prob], {input_image.name: image})
    prob_value = session_outputs[0]
    top_5_indices = np.argsort(prob_value[0])[-5:][::-1]
    synsets = [line.rstrip('\n') for line in open('synset.txt')]
    for i in range(5):
      idx = top_5_indices[i]
      print('%f  %s' % (prob_value[0,idx], synsets[idx]))


if __name__ == '__main__':
  # test_forward()
  test_classify_image()
