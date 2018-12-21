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
e = 1e-7
np.random.seed(1)
node_size = 10  # node size
batch_size = 100 # 100
epoch = 10 # 10
train_data_size = 60000
test_data_size = 10000
learning_rate = 0.01
times_per_epoch = train_data_size / batch_size
times = int(epoch * times_per_epoch)

result = {}
result['test'] = {}
result['test']['sigmoid'] =   np.zeros(epoch)
result['test']['relu'] =      np.zeros(epoch)
result['test']['dropout'] =   np.zeros(epoch)
result['train'] = {}
result['train']['sigmoid'] =   np.zeros(times)
result['train']['relu'] =      np.zeros(times)
result['train']['dropout'] =   np.zeros(times)
result['time'] = {}
dropout_rate = 0.25
node_drop = int(node_size * dropout_rate)


#############
# Functions #
#############

def sigmoid(x):
    return np.where( x < -709, 0.0, 1 / (1 + np.exp(-x)))

def relu(x):
    return np.where( x > 0, x, 0.0)

def dropout(x, is_training):
    if is_training: # todo  内包表記にする
        network['dropouter'] = np.ones(x.shape)
        for i in range(x.shape[0]):
            network['dropouter'][i][np.random.randint(0, node_size - 1, node_drop)] = 0
        return np.where(x > 0, x * network['dropouter'], 0.0)

        # flat = x.shape[0] * x.shape[1]
        # dropout = np.ones(flat,)
        # dropout[np.random.randint(0, flat - 1, node_drop)] = 0
        # network['dropouter'] = dropout.reshape(x.shape)
        # return np.where(x > 0, x * network['dropouter'], 0.0)

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

    network['r'][1], network['r'][2] = 1,1
    network['b'][1], network['b'][2] = 0,0
    network['w'][1] = np.random.normal(0, 1/math.sqrt(784),(784,node_size))
    network['b'][1] = np.random.normal(0, 1/math.sqrt(784),(node_size, ))
    network['w'][2] = np.random.normal(0, 1/math.sqrt(node_size),(node_size,10))
    network['b'][2] = np.random.normal(0, 1/math.sqrt(node_size),(10, ))
    network['batch_avg_sum'][1] = 0
    network['batch_avg_sum'][2] = 0
    network['batch_dsp_sum'][1] = 0
    network['batch_dsp_sum'][2] = 0
    network['counter'][1] = 0
    network['counter'][2] = 0


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

def forward_propagation (input, fun, is_training):

    # input         (input  => x1)
    # => dot        (x1     => y1)
    # => normalize  (y1     => z1)
    # => relu       (z1     => x2)
    # => dot        (x2     => y2)
    # => normalize  (y2     => z2)
    # => softmax    (z2     => output)

    network['x'][1] = input
    network['y'][1] = np.dot(network['x'][1], network['w'][1]) + network['b'][1]
    network['z'][1] = normalization(network['y'][1], 1, is_training)

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
    return          softmax( network['z'][2] )

def back_propagation (output, fun, label):

    # output
    # => softmax    (output     => grad_z2)
    # => normalize  (grad_z2    => grad_y2)
    # => dot        (grad_y2    => grad_x2)
    # => relu       (grad_x2    => grad_z1)
    # => normalize  (grad_z1    => grad_y1)
    # => dot        (grad_y1    => grad_x1)

    grad_z2 = (output - label) / batch_size
    grad_y2 = back_normalization(grad_z2, 2)

    grad_x2 = np.dot(grad_y2, network['w'][2].T)
    network['w'][2] -= learning_rate * np.dot(network['x'][2].T, grad_y2)
    network['b'][2] -= learning_rate * grad_y2.sum(axis=0)

    if fun == 'sigmoid':
        grad_z1 = network['y'][1] * (1 - network['y'][1]) * grad_x2
    elif fun == 'relu':
        grad_z1 = np.where(network['y'][1] > 0, grad_x2, 0.0)
    elif fun == 'dropout':
        grad_z1 = np.where(network['y'][1] > 0, network['dropouter'] * grad_x2, 0.0)
    else:
        return

    grad_y1 = back_normalization(grad_z1, 1)

    network['w'][1] -= learning_rate * np.dot(network['x'][1].T, grad_y1)
    network['b'][1] -= learning_rate * grad_y1.sum(axis=0)
    return 0

def normalization(unnormalized, index, is_training):

    # load
    r = network['r'][index]
    b = network['b'][index]

    if is_training:

        # calculate
        avg = np.sum(unnormalized, axis=0) / unnormalized.shape[0]
        dsp = np.sum(np.power( (unnormalized - avg) , 2), axis=0) / unnormalized.shape[0]
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
    grad_dsp = np.sum(grad_xi * (xi - avg), axis=0) * (-1/2) * np.power(dsp + e, -3/2)
    grad_avg = -np.sum(grad_xi, axis=0) / np.sqrt(dsp + e) \
                   + grad_dsp * (-2) * np.sum(xi - avg, axis=0) / xi.shape[0]

    grad_unnormalized = grad_xi / np.sqrt(dsp + e) \
                        + grad_dsp * 2 * (xi - avg) / xi.shape[0] \
                        + grad_avg / xi.shape[0]

    # update
    network['r'][index] -= learning_rate * np.sum(grad_normalized * _xi)
    network['b'][index] -= np.sum(grad_normalized)

    return grad_unnormalized

def plot(data, label_name = ''):

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

# activation_functions = ['sigmoid', 'relu', 'dropout']
activation_functions = ['relu', 'dropout']

for fun in activation_functions:

    network = init()
    start = time.time()

    # training
    is_training = True
    for i in range(result['train'][fun].size):
        input, label = chose_batch(X, Y)
        output = forward_propagation(input, fun, is_training)
        result['train'][fun][i] = cross_entropy_error(output, label)
        # print(result['train'][fun][i])
        back_propagation(output, fun, label)

        # testing
        if i % times_per_epoch == 0:
            index = int(i / times_per_epoch)
            result['test'][fun][index] = np.sum(forward_propagation(testX, fun, False).argmax(axis=1) == testY.argmax(axis=1)) / test_data_size

    result['time'][fun] = time.time() - start

# print result
print()
for fun in activation_functions:
    print(fun, result['test'][fun])
    print('time : {0}sec'.format(result['time'][fun]))

# np.save('network.npy', network)
# np.load('network.npy')

plt.subplot(211)
plot(result['train']['sigmoid'],    'train:sigmoid')
plot(result['train']['relu'],       'train:relu')
plot(result['train']['dropout'],    'train:dropout')
plt.subplot(212)
plot(result['test']['sigmoid'],     'test:sigmoid')
plot(result['test']['relu'],        'test:relu')
plot(result['test']['dropout'],     'test:dropout')
# プロット表示(設定の反映)
plt.show()
