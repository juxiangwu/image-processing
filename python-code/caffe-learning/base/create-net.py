#coding:utf-8
import caffe
from caffe import layers as L, params as P
def lenet(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec() # 见详解目录-1

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)

    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss =  L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto() #写入到prototxt文件

# with open('D:/Develop/DL/sources/caffe/examples/mnist/lenet_auto_train.prototxt', 'w') as f:
#     f.write(str(lenet('temp/mnist/mnist_data/mnist_train_lmdb', 64)))

# with open('D:/Develop/DL/sources/caffe/examples/mnist/lenet_auto_test.prototxt', 'w') as f:
#     f.write(str(lenet('temp/mnist/mnist_data/mnist_test_lmdb', 100)))

caffe.set_device(0) #选择默认gpu
caffe.set_mode_gpu()    #使用gpu

### load the solver and create train and test nets
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
solver = caffe.SGDSolver('temp/mnist/lenet_auto_solver.prototxt')

result = [(k, v.data.shape) for k, v in solver.net.blobs.items()]
print(result)
print([(k, v[0].data.shape) for k, v in solver.net.params.items()])