#coding:utf-8
import caffe
import numpy as np
import matplotlib.pyplot as plt

im = caffe.io.load_image('datas/cat.jpg')
print(im.shape)
plt.imshow(im)
plt.axis('off')
plt.show()