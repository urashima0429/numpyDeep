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
epoch = 10000 # 10
learning_rate = 0.01
result = np.zeros(epoch)

#############
# Functions #
#############

def sigmoid(x):
    return np.where( x < -709, 0.0, 1 / (1 + np.exp(-x)))

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    return exp_x / np.sum(exp_x)

def cross_entropy_error(x, y):
    return -np.sum(y * np.log(x)) / x.shape[0]

def init():
    network = {}
    network['w1'] = np.random.normal(0, 1/math.sqrt(784),(784,node_size))
    network['b1'] = np.random.normal(0, 1/math.sqrt(784),(node_size, ))
    network['w2'] = np.random.normal(0, 1/math.sqrt(node_size),(node_size,10))
    network['b2'] = np.random.normal(0, 1/math.sqrt(node_size),(10, ))
    return network

def load_data():
    X, Y = mndata.load_training()
    X = np.array(X).reshape((60000, 784)) / 255    # 0 ~ 1 :(60000, 784)
    Y = np.identity(10)[np.array(Y)]          # one-hot vector : (60000, 10)
    return X, Y

def chose_batch(X, Y):
    batch = np.random.randint(0, 59999, batch_size)
    return X[batch], Y[batch]

def forward_propagation (input):
    network['x1'] = input
    network['x2'] = sigmoid( np.dot(network['x1'], network['w1']) + network['b1'] )
    return          softmax( np.dot(network['x2'], network['w2']) + network['b2'] )

def back_propagation (output, label):

    grad_En_Y2 = (output - label) / batch_size

    grad_En_X2 = np.dot(grad_En_Y2, network['w2'].T)
    network['w2'] -= learning_rate * np.dot(network['x2'].T, grad_En_Y2)
    network['b2'] -= learning_rate * grad_En_Y2.sum(axis=0)

    grad_En_Y1 = network['x2'] * (1 - network['x2']) * grad_En_X2
    network['w1'] -= learning_rate * np.dot(network['x1'].T, grad_En_Y1)
    network['b1'] -= learning_rate * grad_En_Y1.sum(axis=0)
    return 0

def plot(result, label_name = ''):

    # データ生成
    x = np.arange(result.size)
    y = result[x]

    # プロット
    plt.plot(x, y, label=label_name)

    # 凡例の表示
    if label_name != '':
        plt.legend()

    # プロット表示(設定の反映)
    plt.show()




########
# main #
########

start = time.time()

network = init()
X, Y = load_data()

for i in range(epoch):
    input, label = chose_batch(X, Y)
    output = forward_propagation(input)
    error = cross_entropy_error(output, label)
    print(error)
    result[i] = error
    print(result[i])
    back_propagation(output, label)

# np.save('network.npy', network)
# np.load('network.npy')

elapsed = time.time() - start
print()
print('time : {0}sec'.format(elapsed))

plot(result, 'error')