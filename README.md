# vgg16.tf
This repo contains a TensorFlow implementation of the VGG16 model (http://arxiv.org/abs/1409.1556), and scripts for converting the pretrained caffemodel into a TensorFlow checkpoint.

**Why another implementation?** Different from the previous implementation at https://github.com/ry/tensorflow-vgg16, this implementaion uses NCHW tensor layout in convolutions, thus should be more efficient. It also allows fine-tuning model parameters by using `tf.Variables` for model parameters instead of `tf.constant`.

## Prerequisites
  1. BVLC Caffe http://caffe.berkeleyvision.org/, needs to build pycaffe (`make pycaffe`)
  2. Python libraries: numpy, skimage. Install them by running `pip install numpy skimage`
  3. TensorFlow https://www.tensorflow.org/, version >= 0.8

## Setup
  1. Download the VGG16 prototxt and caffemodel from https://gist.github.com/ksimonyan/211839e770f7b538e2d8
  2. Modify `run.sh` to set the paths for `--caffe_root`, `--prototxt_path`, and `--caffemodel_path`
  3. Execute `run.sh`

To validate, run `python tests.py --ckpt_path <checkpoint-path>`. Expected output:

    Top 5 predictions:
    0.998706  n02123159 tiger cat
    0.001294  n02124075 Egyptian cat
    0.000000  n02441942 weasel
    0.000000  n02127052 lynx, catamount
    0.000000  n02123045 tabby, tabby cat

In some cases, you may need another scope name for the model, change `--model_scope` to set the model scope name.
