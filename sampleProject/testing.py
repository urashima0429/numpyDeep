import time
import math
import numpy as np
from mnist import MNIST
import pickle
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided

#############
# Constants #
#############

# data imformation
mndata = MNIST("/export/home/016/a0165336/project/le4nn/")

###############
# change me!! #
###############
node_size = 50  # node size
batch_size = 100  # 100
epoch = 3  # 10
e = 1e-10
dropout_rate = 0.1
node_drop = int(node_size * dropout_rate)

# convolve
filter_size = 5
filter_num = 4
channel_num = 3
padding_size = int(filter_size / 2)
input_x = 28
input_y = 28

# SGD
sgd = {
    'learning_rate': 0.01
}

# Momentum SGD
msgd = {
    1: {
        'deltaW': 0
    },
    2: {
        'deltaW': 0
    },
    'learning_rate': 0.01,
    'momentum_rate': 0.9
}

# AdaGrad
adagrad = {
    1: {
        'h': 1e-8
    },
    2: {
        'h': 1e-8
    },
    'learning_rate': 0.001
}

# RMSProp
rmsprop = {
    1: {
        'h': 0
    },
    2: {
        'h': 0
    },
    'learning_rate': 0.001,
    'rho': 0.9,
    'e': 1e-10
}

# AdaDelta
adadelta = {
    1: {
        'h': 0,
        's': 0
    },
    2: {
        'h': 0,
        's': 0
    },
    'rho': 0.95,
    'e': 1e-6
}

# Adam
adam = {
    1: {
        't': 0,
        'm': 0,
        'v': 0
    },
    2: {
        't': 0,
        'm': 0,
        'v': 0
    },
    'alpha': 0.001,
    'beta1': 0.9,
    'beta2': 0.999,
    'e': 1e-8
}

# set my constants
np.random.seed(0)
# train_data_size = 50000
train_data_size = 60000
test_data_size = 10000
# vector_size = 32*32*3
vector_size = 28*28
label_size = 10
data_max = 255
times_per_epoch = train_data_size / batch_size
times = int(epoch * times_per_epoch)

result = {}
result['time'] = {}
result['test'] = {}
result['train'] = {}

activation_functions = ['sigmoid', 'relu', 'dropout']
for fun in activation_functions:
    result['train'][fun] = np.zeros(times)
    result['test'][fun] = np.zeros(epoch)

optimization_methods = ['SGD','Adam']
# optimization_methods = ['SGD', 'MSGD', 'AdaGrad', 'RMSProp', 'AdaDelta', 'Adam']
for method in optimization_methods:
    result['train'][method] = np.zeros(times)
    result['test'][method] = np.zeros(epoch)




#############
# Functions #
#############

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.where(x > 0, x, 0.0)

def dropout(x, is_training):
    if is_training:  # todo  内包表記にする
        network['dropouter'] = np.ones(x.shape)
        for i in range(x.shape[0]):
            network['dropouter'][i][np.random.randint(0, node_size - 1, node_drop)] = 0
        return np.where(x > 0, x * network['dropouter'], 0.0)

    else:
        return np.where(x > 0, x * (1 - dropout_rate), 0.0)

def softmax(x):
    c = np.max(x, axis=1)  # todo
    exp_x = np.exp((x.T - c).T)
    return (exp_x.T / np.sum(exp_x, axis=1)).T  # todo

def cross_entropy_error(x, y):
    return -np.sum(y * np.log(x)) / x.shape[0]

# (batch_size, channel_num, input_x, input_y) => (batch_size, channel_num * input_x * input_y * filter_num)
def convolve(x):
    ps = padding_size
    fs = filter_size

    # (img_num, 28, 28) => (img_num, 28 + ps, 28 + ps)
    padded = np.pad(x, [(0, 0), (0, 0), (ps, ps), (ps, ps)], 'constant').transpose(0,2,3,1)

    # as_strided によって padded を stride 幅で (28,28,3,3)をきりだす
    # [0   0   0]
    # [0   0   1]
    # [0  28  29]] ...
    # submatrices = as_strided(padded, (padded.shape[0],28,28,1,fs,fs), (padded.shape[0],240, 8, padded.shape[0],240, 8))
    submatrices = as_strided(padded, (batch_size, input_x, input_y, channel_num, fs, fs), (padded.shape[1] * padded.shape[2] * 8, padded.shape[1] * 8, 8, padded.shape[0] * padded.shape[1] * padded.shape[2] * 8, padded.shape[2] * 8, 8))
    # print(submatrices.shape)

    # (batch_size, input_x, input_y, fs, fs) => (input_x * input_y * batch_size , fs*fs*channel_num).T
    x = submatrices.reshape(batch_size * input_x * input_y, channel_num * fs * fs).T

    print(x.T[0].reshape(channel_num,filter_size,filter_size))
    print(x.T[-1].reshape(channel_num,filter_size,filter_size))

    y = np.dot(network['cw'], x) + network['cb']

    return y.reshape(filter_num * input_x * input_y, batch_size).T

# (batch_size, channel_num * input_x * input_y * filter_num) => (batch_size, channel_num, input_x, input_y)
def back_convolve(dy):
    # ps = padding_size
    # fs = filter_size
    #
    # # (img_num, 28, 28) => (img_num, 28 + ps, 28 + ps)
    # padded = np.pad(x, [(0, 0), (0, 0), (ps, ps), (ps, ps)], 'constant').transpose(0,2,3,1)
    #
    # # as_strided によって padded を stride 幅で (28,28,3,3)をきりだす
    # # [0   0   0]
    # # [0   0   1]
    # # [0  28  29]] ...
    # # submatrices = as_strided(padded, (padded.shape[0],28,28,1,fs,fs), (padded.shape[0],240, 8, padded.shape[0],240, 8))
    # submatrices = as_strided(padded, (batch_size, input_x, input_y, channel_num, fs, fs), (padded.shape[1] * padded.shape[2] * 8, padded.shape[1] * 8, 8, padded.shape[0] * padded.shape[1] * padded.shape[2] * 8, padded.shape[2] * 8, 8))
    # print(submatrices.shape)
    #
    # # (batch_size, input_x, input_y, fs, fs) => (input_x * input_y * batch_size , fs*fs*channel_num).T
    # x = submatrices.reshape(batch_size * input_x * input_y, channel_num * fs * fs).T
    #
    # print(x.T[0].reshape(channel_num,filter_size,filter_size))
    # print(x.T[-1].reshape(channel_num,filter_size,filter_size))
    #
    # y = np.dot(network['cw'], x) + network['cb']
    #
    # return y.reshape(filter_num * input_x * input_y, batch_size).T


    dy = dy.reshape(filter_num, input_x * input_y * batch_size)
    dx = np.dot(network['cw'].T, dy)
    # print(dx.shape)

    # dw = np.dot(dy, network['x'][1])



    return

def pooling(a):
    return a

def back_pooling(a):
    return a

def init():
    network = {}
    network['w'] = {}
    network['b'] = {}
    network['x'] = {}
    network['y'] = {}
    network['z'] = {}
    network['r'] = {}
    network['b'] = {}
    network['batch_avg'] = {}
    network['batch_dsp'] = {}
    network['batch_normalized_xi'] = {}
    network['batch_avg_sum'] = {}
    network['batch_dsp_sum'] = {}
    network['counter'] = {}

    network['r'][1], network['r'][2] = 1, 1
    network['b'][1], network['b'][2] = 0, 0
    network['w'][1] = np.random.normal(0, 1 / math.sqrt(vector_size), (vector_size, node_size))
    network['b'][1] = np.random.normal(0, 1 / math.sqrt(vector_size), (node_size,))
    network['w'][2] = np.random.normal(0, 1 / math.sqrt(node_size), (node_size, label_size))
    network['b'][2] = np.random.normal(0, 1 / math.sqrt(node_size), (label_size,))

    network['cw'] = np.random.normal(0, 1 / math.sqrt(10), (filter_num, filter_size * filter_size * channel_num))
    network['cb'] = np.random.normal(0, 1 / math.sqrt(10), (filter_num, input_x * input_y * batch_size))

    network['batch_avg_sum'][1] = 0
    network['batch_avg_sum'][2] = 0
    network['batch_dsp_sum'][1] = 0
    network['batch_dsp_sum'][2] = 0
    network['counter'][1] = 0
    network['counter'][2] = 0

    return network

# (train_data_size, vector_size), (train_data_size, label_size)
def load_train_data():
    X, Y = mndata.load_training()
    X = np.array(X).reshape((train_data_size, input_x * input_y)) / data_max  # 0 ~ 1 :(60000, 784)
    Y = np.identity(label_size)[np.array(Y)]  # one-hot vector : (60000, 10)
    return X, Y

# (train_data_size, vector_size), (train_data_size, label_size)
def load_test_data():
    X, Y = mndata.load_testing()
    X = np.array(X).reshape((test_data_size, input_x * input_y)) / data_max  # 0 ~ 1 :(60000, 784)
    Y = np.identity(label_size)[np.array(Y)]  # one-hot vector : (60000, 10)
    return X, Y

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# # (train_data_size, vector_size), (train_data_size, label_size)
# def load_train_data():
#     train_data = {}
#     for i in range(5):  # 1~5
#         train_data[i] = unpickle("/export/home/016/a0165336/project/le4nn/cifar-10-batches-py/data_batch_{0}".format(i + 1))
#
#     X = np.vstack((
#         train_data[0][b'data'],
#         train_data[1][b'data'],
#         train_data[2][b'data'],
#         train_data[3][b'data'],
#         train_data[4][b'data']
#     )) / data_max
#     Y = np.identity(label_size)[np.hstack((
#         train_data[0][b'labels'],
#         train_data[1][b'labels'],
#         train_data[2][b'labels'],
#         train_data[3][b'labels'],
#         train_data[4][b'labels']
#     ))]
#     return X, Y
#
# # (train_data_size, vector_size), (train_data_size, label_size)
# def load_test_data():
#     test_data = unpickle("/export/home/016/a0165336/project/le4nn/cifar-10-batches-py/test_batch")
#     X = test_data[b'data'] / data_max
#     Y = np.identity(label_size)[test_data[b'labels']]
#     return X, Y

def chose_batch(X, Y):
    batch = np.random.randint(0, train_data_size - 1, batch_size)
    return X[batch], Y[batch]

def forward_propagation(input, fun, is_training):
    # input         (input  => x1)
    # => dot        (x1     => y1)
    # => normalize  (y1     => z1)
    # => relu       (z1     => x2)
    # => dot        (x2     => y2)
    # => normalize  (y2     => z2)
    # => softmax    (z2     => output)

    # network['input'] = input
    #
    # conv = convolve(network['input'])
    # if fun == 'sigmoid':
    #     network['conv'] = sigmoid(conv)
    # elif fun == 'relu':
    #     network['conv'] = relu(conv)
    # elif fun == 'dropout':
    #     network['conv'] = dropout(conv, is_training)
    # else:
    #     return
    #
    # pool = pooling(network['conv'])
    # if fun == 'sigmoid':
    #     network['pool'] = sigmoid(pool)
    # elif fun == 'relu':
    #     network['pool'] = relu(pool)
    # elif fun == 'dropout':
    #     network['pool'] = dropout(pool, is_training)
    # else:
    #     return
    #
    # network['x'][1] = network['pool']


    network['x'][1] = input
    network['y'][1] = np.dot(network['x'][1], network['w'][1]) + network['b'][1]
    network['z'][1] = normalization(network['y'][1], 1, is_training)
    # network['z'][1] = network['y'][1]


    if fun == 'sigmoid':
        network['x'][2] = sigmoid(network['z'][1])
    elif fun == 'relu':
        network['x'][2] = relu(network['z'][1])
    elif fun == 'dropout':
        network['x'][2] = dropout(network['z'][1], is_training)
    else:
        return

    network['y'][2] = np.dot(network['x'][2], network['w'][2]) + network['b'][2]
    network['z'][2] = normalization(network['y'][2], 2, is_training)


    # network['z'][2] = network['y'][2]

    return softmax(network['z'][2])


def back_propagation(output, fun, label):
    # output
    # => softmax    (output     => grad_z2)
    # => normalize  (grad_z2    => grad_y2)
    # => dot        (grad_y2    => grad_x2)
    # => relu       (grad_x2    => grad_z1)
    # => normalize  (grad_z1    => grad_y1)
    # => dot        (grad_y1    => grad_x1)

    grad_z2 = (output - label) / batch_size
    grad_y2 = back_normalization(grad_z2, 2)
    # grad_y2 = grad_z2

    grad_x2 = np.dot(grad_y2, network['w'][2].T)

    deltaW = np.dot(network['x'][2].T, grad_y2)
    deltaB = grad_y2.sum(axis=0)
    network['w'][2] += optimizer(deltaW, method, 2)
    network['b'][2] += optimizer(deltaB, 'SGD', 2)

    if fun == 'sigmoid':
        grad_z1 = grad_x2 * (1 - grad_x2)
    elif fun == 'relu':
        grad_z1 = np.where(network['y'][1] > 0, grad_x2, 0.0)
    elif fun == 'dropout':
        grad_z1 = np.where(network['y'][1] > 0, network['dropouter'] * grad_x2, 0.0)
    else:
        return

    grad_y1 = back_normalization(grad_z1, 1)
    # grad_y1 = grad_z1

    deltaW = np.dot(network['x'][1].T, grad_y1)
    deltaB = grad_y1.sum(axis=0)
    network['w'][1] += optimizer(deltaW, method, 1)
    network['b'][1] += optimizer(deltaB, 'SGD', 1)
    return 0


def normalization(unnormalized, index, is_training):
    # load
    r = network['r'][index]
    b = network['b'][index]

    if is_training:

        # calculate
        avg = np.sum(unnormalized, axis=0) / unnormalized.shape[0]
        dsp = np.sum(np.power((unnormalized - avg), 2), axis=0) / unnormalized.shape[0]
        xi = (unnormalized - avg) / np.sqrt(dsp + e)
        normalized = xi * r + b

        # update
        network['batch_avg'][index] = avg
        network['batch_dsp'][index] = dsp
        network['batch_normalized_xi'][index] = xi
        network['batch_avg_sum'][index] += avg
        network['batch_dsp_sum'][index] += dsp
        network['counter'][index] += 1

    else:
        # calculate
        dsp_avg = network['batch_dsp_sum'][index] / network['counter'][index]
        avg_avg = network['batch_avg_sum'][index] / network['counter'][index]
        std_avg = np.sqrt(dsp_avg + e)
        normalized = r * (unnormalized - avg_avg) / std_avg + b

    return normalized


def back_normalization(grad_normalized, index):
    # load
    r = network['r'][index]
    xi = network['y'][index]
    _xi = network['batch_normalized_xi'][index]
    avg = network['batch_avg'][index]
    dsp = network['batch_dsp'][index]

    # calculate
    grad_xi = grad_normalized * r
    grad_dsp = np.sum(grad_xi * (xi - avg), axis=0) * (-1 / 2) * np.power(dsp + e, -3 / 2)
    grad_avg = -np.sum(grad_xi, axis=0) / np.sqrt(dsp + e) \
               + grad_dsp * (-2) * np.sum(xi - avg, axis=0) / xi.shape[0]

    grad_unnormalized = grad_xi / np.sqrt(dsp + e) \
                        + grad_dsp * 2 * (xi - avg) / xi.shape[0] \
                        + grad_avg / xi.shape[0]

    # update
    deltaR = np.sum(grad_normalized * _xi, axis=0)
    deltaB = np.sum(grad_normalized, axis=0)
    network['r'][index] += optimizer(deltaR, 'SGD', index)  # normalize ha SGD ni kotei
    network['b'][index] += optimizer(deltaB, 'SGD', index)  # normalize ha SGD ni kotei

    return grad_unnormalized


def optimizer(grad, method, index):
    if method == 'SGD':
        optimizer = -sgd['learning_rate'] * grad

    elif method == 'MSGD':
        tmp = msgd['momentum_rate'] * msgd[index]['deltaW'] - msgd['learning_rate'] * grad
        msgd[index]['deltaW'] = tmp
        optimizer = tmp

    elif method == 'AdaGrad':
        tmp = adagrad[index]['h'] + grad * grad
        adagrad[index]['h'] = tmp
        optimizer = -adagrad['learning_rate'] * grad / np.sqrt(tmp)

    elif method == 'RMSProp':
        rho = rmsprop['rho']
        e = rmsprop['e']
        h = rmsprop[index]['h']

        tmp = rho * h + (1 - rho) * grad * grad
        rmsprop[index]['h'] = tmp
        optimizer = -rmsprop['learning_rate'] * grad / np.sqrt(tmp + e)

    elif method == 'AdaDelta':
        rho = adadelta['rho']
        e = adadelta['e']
        h = adadelta[index]['h']
        s = adadelta[index]['s']

        adadelta[index]['h'] = rho * h + (1 - rho) * grad * grad
        deltaW = -grad * np.sqrt(s + e) / np.sqrt(h + e)
        adadelta[index]['s'] = rho * s + (1 - rho) * deltaW * deltaW
        optimizer = deltaW

    elif method == 'Adam':

        adam[index]['t'] = adam[index]['t'] + 1
        adam[index]['m'] = adam['beta1'] * adam[index]['m'] + (1 - adam['beta1']) * grad
        adam[index]['v'] = adam['beta2'] * adam[index]['v'] + (1 - adam['beta2']) * grad * grad
        _m = adam[index]['m'] / (1 - np.power(adam['beta1'], adam[index]['t']))
        _v = adam[index]['v'] / (1 - np.power(adam['beta2'], adam[index]['t']))
        optimizer = -adam['alpha'] * _m / (np.sqrt(_v) + adam['e'])
    else:
        optimizer = 0

    return optimizer

def plot(data, label_name=''):
    # データ生成
    x = np.arange(data.size)
    y = data[x]

    # プロット
    plt.plot(x, y, label=label_name)

    # 凡例の表示
    if label_name != '':
        plt.legend()

########
# main #
########

X, Y = load_train_data()
testX, testY = load_test_data()

fun = 'dropout'
for method in optimization_methods:

    network = init()

    # test
    a = np.arange(batch_size * channel_num * input_x * input_y).reshape(batch_size, channel_num, input_x, input_y)
    b = convolve(a)

    c = back_convolve(b)

    #

    start = time.time()


    # training
    for i in range(result['train'][method].size):
        input, label = chose_batch(X, Y)
        output = forward_propagation(input, fun, True)
        result['train'][method][i] = cross_entropy_error(output, label)
        # print(result['train'][fun][i])
        back_propagation(output, fun, label)

        # testing
        if i % times_per_epoch == 0:
            index = int(i / times_per_epoch)
            result['test'][method][index] = np.sum(
                forward_propagation(testX, fun, False).argmax(axis=1) == testY.argmax(axis=1)) / test_data_size

    result['time'][method] = time.time() - start

# print result
print()
for method in optimization_methods:
    print(method)
    print('time : {0}sec'.format(result['time'][method]))
    print(result['test'][method])
    print('accuracity average : {0}'.format( np.average(result['test'][method]) ) )
    print()

    plot(result['test'][method], method)

# np.save('network.npy', network)
# np.load('network.npy')
# プロット表示(設定の反映)
plt.show()