#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import pylab
from PIL import Image
import caffe
import cv2

caffe_root = "../resources/models/caffe/bvlc_googlenet/"
net_file = caffe_root + "deploy.prototxt"
model = caffe_root + "bvlc_googlenet.caffemodel"

caffe.set_mode_gpu()
net = caffe.Net(net_file, model, caffe.TEST)
image_mean = np.load("../resources/models/caffe/ilsvrc_2012_mean.npy").mean(1).mean(1)

# print 'mean-subtracted values:', zip('RGB', image_mean)
# 输出结果：mean-subtracted values: [('R', 104.0069879317889), ('G', 116.66876761696767), ('B', 122.6789143406786)]

data_shape = net.blobs['data'].data.shape
# print data_shape
# 输出结果：(10, 3, 224, 224) batch_size:10  channels:3  height:224  weight:224

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', image_mean)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))

image = caffe.io.load_image('../resources/images/dog-cycle-car.png')
transformed_image = transformer.preprocess('data', image)
# plt.imshow(image)
# pylab.show()

net.blobs['data'].data[...] = transformed_image
output = net.forward()
print(output['prob'])
print(output['prob'].shape)
output_prob = output['prob'][0]
print('The predicted class is : ', output_prob.argmax())

label_file = "../resources/models/caffe/synset_words.txt"
labels = np.loadtxt(label_file, str, delimiter='\t')
print('The label is : ', labels[output_prob.argmax()])

top_inds = output_prob.argsort()[::-1][:5]
print('probabilities and labels: ')
ziped = zip(output_prob[top_inds], labels[top_inds])
for res in ziped:
    print(res)