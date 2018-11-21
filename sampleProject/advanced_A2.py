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
epoch = 10 # 10
train_data_size = 60000
test_data_size = 10000
learning_rate = 0.01
train_result = np.zeros( int(epoch * train_data_size / batch_size) )
is_training = False
dropout_rate = 0.2
node_drop = int(node_size * dropout_rate)


#############
# Functions #
#############

# def sigmoid(x):
#     return np.where( x < -709, 0.0, 1 / (1 + np.exp(-x)))

def relu(x):
    return np.where( x > 0, x, 0.0)

def dropout(x, is_training):
    if is_training: # todo  内包表記にする
        network['dropouter'] = np.ones(x.shape)
        for i in range(x.shape[0]):
            network['dropouter'][i][np.random.randint(0, node_size - 1, node_drop)] = 0
        print(network['dropouter'])
        return np.where(x > 0, x * network['dropouter'], 0.0)

    else:
        return np.where(x > 0, x * (1-dropout_rate), 0.0)


def softmax(x):
    c = np.max(x, axis=1) # todo
    exp_x = np.exp( (x.T - c).T )
    return (exp_x.T / np.sum(exp_x, axis=1)).T # todo

def cross_entropy_error(x, y):
    return -np.sum(y * np.log(x)) / x.shape[0]

def init():
    network = {}
    network['w1'] = np.random.normal(0, 1/math.sqrt(784),(784,node_size))
    network['b1'] = np.random.normal(0, 1/math.sqrt(784),(node_size, ))
    network['w2'] = np.random.normal(0, 1/math.sqrt(node_size),(node_size,10))
    network['b2'] = np.random.normal(0, 1/math.sqrt(node_size),(10, ))
    return network

def load_train_data():
    X, Y = mndata.load_training()
    X = np.array(X).reshape((train_data_size, 784)) / 255    # 0 ~ 1 :(60000, 784)
    Y = np.identity(10)[np.array(Y)]          # one-hot vector : (60000, 10)
    return X, Y

def load_test_data():
    X, Y = mndata.load_testing()
    X = np.array(X).reshape((test_data_size, 784)) / 255    # 0 ~ 1 :(60000, 784)
    Y = np.identity(10)[np.array(Y)]          # one-hot vector : (60000, 10)
    return X, Y

def chose_batch(X, Y):
    batch = np.random.randint(0, 59999, batch_size)
    return X[batch], Y[batch]

def forward_propagation (input, is_training):
    network['x1'] = input
    network['y1'] = np.dot(network['x1'], network['w1']) + network['b1']
    # network['x2'] = relu(network['y1'])
    network['x2'] = dropout(network['y1'], is_training)
    network['x2'] = relu(network['y1'])


    network['y2'] = np.dot(network['x2'], network['w2']) + network['b2']
    return          softmax( network['y2'] )

def back_propagation (output, label):

    grad_En_Y2 = (output - label) / batch_size

    grad_En_X2 = np.dot(grad_En_Y2, network['w2'].T)
    network['w2'] -= learning_rate * np.dot(network['x2'].T, grad_En_Y2)
    network['b2'] -= learning_rate * grad_En_Y2.sum(axis=0)

    # sigmoid
    # grad_En_Y1 = network['x2'] * (1 - network['x2']) * grad_En_X2

    # relu
    # grad_En_Y1 = np.where(network['y1'] > 0, grad_En_X2, 0.0)

    # dropout
    grad_En_Y1 = np.where(network['y1'] > 0 , network['dropouter'] * grad_En_X2 , 0.0)
    network['w1'] -= learning_rate * np.dot(network['x1'].T, grad_En_Y1)
    network['b1'] -= learning_rate * grad_En_Y1.sum(axis=0)
    return 0

def plot(train_result, label_name = ''):

    # データ生成
    x = np.arange(train_result.size)
    y = train_result[x]

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
X, Y = load_train_data()


# training
is_training = True
for i in range(train_result.size):
    input, label = chose_batch(X, Y)
    output = forward_propagation(input, is_training)
    train_result[i] = cross_entropy_error(output, label)
    print(train_result[i])
    back_propagation(output, label)

# testing
is_training = False
testX, testY = load_test_data()
print(np.sum(forward_propagation(testX, is_training).argmax(axis=1) == testY.argmax(axis=1)) / test_data_size)

# np.save('network.npy', network)
# np.load('network.npy')



elapsed = time.time() - start
print()
print('time : {0}sec'.format(elapsed))

plot(train_result, 'train error')