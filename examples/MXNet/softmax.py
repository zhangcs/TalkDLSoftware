# Fetch the MNIST dataset
import numpy as np
import os, gzip, struct
def read_data(label_url, image_url):
    with gzip.open(label_url) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(image_url, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)

train_lbl, train_img = read_data('train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz')
test_lbl, test_img = read_data('t10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')

# create data iterators for MXNet
import mxnet as mx
def to4d(img):
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255
batch_size = 100
train_iter = mx.io.NDArrayIter(to4d(train_img), train_lbl, batch_size, shuffle=True)
test_iter = mx.io.NDArrayIter(to4d(test_img), test_lbl, batch_size)

# Define the network
import logging
logging.getLogger().setLevel(logging.INFO)

data = mx.sym.Variable('data') # Create a place holder variable for the input data
data = mx.sym.Flatten(data=data)# Flatten the data from 4-D shape into 2-D
fc1  = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128) 
act1 = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")
fc2  = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden = 64)
act2 = mx.sym.Activation(data=fc2, name='relu2', act_type="relu")
fc3  = mx.sym.FullyConnected(data=act2, name='fc3', num_hidden=10)
out  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
mod = mx.mod.Module(out)

# Plot the network graph
mx.viz.plot_network(symbol=out, shape={'data': (batch_size, 1, 28, 28)}).view()

# Train the network
import logging
logging.getLogger().setLevel(logging.INFO)

mod.fit(train_data=train_iter,
        eval_data=test_iter,
        num_epoch=10)
#mod.fit(train_data=train_iter, num_epoch=10)





