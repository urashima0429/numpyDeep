import time
import math
import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from pylab import cm

#############
# Constants #
#############

# data imformation
mndata = MNIST("/export/home/016/a0165336/project/le4nn/")

# set my constants
np.random.seed(0)
node_size = 10  # node size
batch_size = 100 # 100
train_data_size = 60000

#############
# Functions #
#############

def init():
    network = {}
    network['w1'] = np.random.normal(0, 1/math.sqrt(784),(784,node_size))
    network['b1'] = np.random.normal(0, 1/math.sqrt(784),(node_size, ))
    network['w2'] = np.random.normal(0, 1/math.sqrt(node_size),(node_size,10))
    network['b2'] = np.random.normal(0, 1/math.sqrt(node_size),(10, ))
    return network

def sigmoid(x):
    return np.where( x < -709, 0.0, 1 / (1 + np.exp(-x)))

def softmax(x):
    c = np.max(x, axis=1) # todo
    exp_x = np.exp( (x.T - c).T )
    return (exp_x.T / np.sum(exp_x, axis=1)).T # todo

def cross_entropy_error(x, y):
    return -np.sum(y * np.log(x)) / x.shape[0]

def load_train_data():
    X, Y = mndata.load_training()
    X = np.array(X).reshape((train_data_size, 784)) / 255    # 0 ~ 1 :(60000, 784)
    Y = np.identity(10)[np.array(Y)]          # one-hot vector : (60000, 10)
    return X, Y

def chose_batch(X, Y):
    batch = np.random.randint(0, 59999, batch_size)
    return X[batch], Y[batch]

def forward_propagation (input):
    network['x1'] = input
    network['x2'] = sigmoid( np.dot(network['x1'], network['w1']) + network['b1'] )
    return          softmax( np.dot(network['x2'], network['w2']) + network['b2'] )


########
# main #
########

start = time.time()

network = init()
X, Y = load_train_data()

input, label = chose_batch(X, Y)
output = forward_propagation(input)
print(cross_entropy_error(output, label))

elapsed = time.time() - start
print()
print('time : {0}sec'.format(elapsed))

