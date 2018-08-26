#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import caffe

def show_feature(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data)
    plt.axis('off')
    plt.show()

    

plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

net = caffe.Net('../training-results/caffe/mnist/lenet/lenet.prototxt',
                '../training-results/caffe/mnist/lenet/snapshot_iter_10000.caffemodel',
                caffe.TEST)



params = [(k, v[0].data.shape) for k, v in net.params.items()]
print(params)

weight = net.params["conv1"][0].data
print(weight.shape)
show_feature(weight.transpose(0, 2, 3, 1))
plt.show()