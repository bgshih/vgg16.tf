import sys, os
import numpy as np
import joblib

os.environ["GLOG_minloglevel"] = "2"
sys.path.append('/home/bgshi/research/common/caffe/python/')
import caffe


def dump_caffemodel_weights():
  model_root = '/var/share/models/caffe/vgg16/'
  prototxt_path = os.path.join(model_root, 'train_val.prototxt')
  model_path = os.path.join(model_root, 'VGG_ILSVRC_16_layers.caffemodel')
  net = caffe.Net(prototxt_path, model_path, caffe.TEST)

  weights = {}
  n_layers = len(net.layers)
  for i in range(n_layers):
    layer_name = net._layer_names[i]
    layer = net.layers[i]
    layer_blobs = [o.data for o in layer.blobs]
    weights[layer_name] = layer_blobs
  joblib.dump(weights, '../data/VGG_ILSVRC_16_layers.pkl')

if __name__ == '__main__':
  dump_caffemodel_weights()
