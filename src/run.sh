# convert caffe model
python dump_caffemodel_weights.py \
  --caffe_root /home/bgshi/research/common/caffe/ \
  --prototxt_path /var/share/models/caffe/vgg16/train_val.prototxt \
  --caffemodel_path /var/share/models/caffe/vgg16/VGG_ILSVRC_16_layers.caffemodel

python convert_caffemodel_to_ckpt.py \
  --model_scope Vgg16 \
  --ckpt_path ./VGG_ILSVRC_16_layers.ckpt
