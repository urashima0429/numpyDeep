import time
import math
import numpy as np
from mnist import MNIST
import pickle
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided

class DataLoader:

    def __init__(self, dataset):
        self.dataset = dataset

        # set input image info
        if dataset == 'mnist':
            # mnist
            self.mndata = MNIST("/export/home/016/a0165336/project/le4nn/")
            self.train_data_size = 60000
            self.test_data_size = 10000
            self.data_max = 255
            self.label_size = 10
            self.input_x = 28
            self.input_y = 28
            self.input_channel_num = 1

        elif dataset == 'cifer':
            # cifer-10
            self.train_data_address = '/export/home/016/a0165336/project/le4nn/cifar-10-batches-py/data_batch_{0}'
            self.test_data_address  = '/export/home/016/a0165336/project/le4nn/cifar-10-batches-py/test_batch'
            self.train_data_size = 50000
            self.test_data_size = 10000
            self.data_max = 255
            self.label_size = 10
            self.input_x = 32
            self.input_y = 32
            self.input_channel_num = 3

        else:
            return

    def _unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    # (train_data_size, vector_size), (train_data_size, label_size)
    def load_train_data(self):
        if self.dataset == 'mnist':
            X, Y = self.mndata.load_training()
            # X = np.array(X).reshape((train_data_size, input_channel_num, input_x, input_y)) / data_max  # 0 ~ 1 :(60000, 784)
            X = np.array(X).reshape(
                (self.train_data_size, self.input_channel_num , self.input_x , self.input_y)) / self.data_max  # 0 ~ 1 :(60000, 784)
            Y = np.identity(self.label_size)[np.array(Y)]  # one-hot vector : (60000, 10)
            return X, Y

        elif self.dataset == 'cifer':
            train_data = {}
            for i in range(5):  # 1~5
                train_data[i] = self._unpickle(self.train_data_address.format(i + 1))

            X = np.vstack((
                train_data[0][b'data'],
                train_data[1][b'data'],
                train_data[2][b'data'],
                train_data[3][b'data'],
                train_data[4][b'data']
            )).reshape((self.train_data_size, self.input_channel_num , self.input_x , self.input_y)) / self.data_max
            Y = np.identity(self.label_size)[np.hstack((
                train_data[0][b'labels'],
                train_data[1][b'labels'],
                train_data[2][b'labels'],
                train_data[3][b'labels'],
                train_data[4][b'labels']
            ))]
            return X, Y

    # (train_data_size, vector_size), (train_data_size, label_size)
    def load_test_data(self):
        if self.dataset == 'mnist':
            X, Y = self.mndata.load_testing()
            # X = np.array(X).reshape((test_data_size, input_channel_num, input_x, input_y)) / data_max  # 0 ~ 1 :(60000, 784)
            X = np.array(X).reshape(
                (self.test_data_size, self.input_channel_num , self.input_x , self.input_y)) / self.data_max  # 0 ~ 1 :(60000, 784)
            Y = np.identity(self.label_size)[np.array(Y)]  # one-hot vector : (60000, 10)
            return X, Y
        elif self.dataset == 'cifer':
            test_data = self._unpickle(self.test_data_address)
            X = test_data[b'data'].reshape((self.test_data_size, self.input_channel_num , self.input_x , self.input_y)) / self.data_max
            Y = np.identity(self.label_size)[test_data[b'labels']]
            return X, Y

class Optimizer:
    # SGD
    sgd = {
        'LEARNING_RATE': 0.01
    }

    # Momentum SGD
    msgd = {
        'deltaW': 0,
        'LEARNING_RATE': 0.01,
        'MOMENTUM_RATE': 0.9
    }

    # AdaGrad
    adagrad = {
        'h': 1e-8,
        'LEARNING_RATE': 0.001
    }

    # RMSProp
    rmsprop = {
        'h': 0,
        'LEARNING_RATE': 0.001,
        'RHO': 0.9,
        'E': 1e-10
    }

    # AdaDelta
    adadelta = {
        'h': 0,
        's': 0,
        'RHO': 0.95,
        'E': 1e-6
    }

    # Adam
    adam = {
        't': 0,
        'm': 0,
        'v': 0,
        'ALPHA': 0.001,
        'BETA1': 0.9,
        'BETA2': 0.999,
        'E': 1e-8
    }

    def __init__(self, method):
        self.method = method
        if self.method == 'SGD': pass
        elif self.method == 'MSGD':     self.deltaW = self.msgd['deltaW']
        elif self.method == 'AdaGrad':  self.h = self.adagrad['h']
        elif self.method == 'RMSProp':  self.h = self.rmsprop['h']
        elif self.method == 'AdaDelta': self.h, self.s = self.adadelta['h'],self.adadelta['s']
        elif self.method == 'Adam':     self.t, self.m, self.v = self.adam['t'], self.adam['m'], self.adam['v']

    def pop(self, grad):
        if self.method == 'SGD':
            optimizer = -self.sgd['LEARNING_RATE'] * grad

        elif self.method == 'MSGD':
            tmp = self.msgd['MOMENTUM_RATE'] * self.deltaW - self.msgd['LEARNING_RATE'] * grad
            self.deltaW = tmp
            optimizer = tmp

        elif self.method == 'AdaGrad':
            tmp = self.h + grad * grad
            self.h = tmp
            optimizer = -self.adagrad['LEARNING_RATE'] * grad / np.sqrt(tmp)

        elif self.method == 'RMSProp':
            rho = self.rmsprop['RHO']
            e = self.rmsprop['E']

            tmp = rho * self.h + (1 - rho) * grad * grad
            self.h = tmp
            optimizer = -self.rmsprop['LEARNING_RATE'] * grad / np.sqrt(tmp + e)

        elif self.method == 'AdaDelta':
            rho = self.adadelta['RHO']
            e = self.adadelta['E']

            self.h = rho * self.h + (1 - rho) * grad * grad
            deltaW = -grad * np.sqrt(self.s + e) / np.sqrt(self.h + e)
            self.s = rho * self.s + (1 - rho) * deltaW * deltaW
            optimizer = deltaW

        elif self.method == 'Adam':

            self.t = self.t + 1
            self.m = self.adam['BETA1'] * self.m + (1 - self.adam['BETA1']) * grad
            self.v = self.adam['BETA2'] * self.v + (1 - self.adam['BETA2']) * grad * grad
            _m = self.m / (1 - np.power(self.adam['BETA1'], self.t))
            _v = self.v / (1 - np.power(self.adam['BETA2'], self.t))
            optimizer = -self.adam['ALPHA'] * _m / (np.sqrt(_v) + self.adam['E'])

        return optimizer

class BatchNormalizer:
    def __init__(self):
        self.r = 1
        self.b = 0
        self.optimizer_r = Optimizer('Adam')
        self.optimizer_b = Optimizer('Adam')
        self.batch_avg = 0
        self.batch_dsp = 0
        self.batch_normalized_xi = 0
        self.batch_avg_sum = 0
        self.batch_dsp_sum = 0
        self.counter = 0

    def forward(self, unnormalized, it):
        if it:
            self.unnormalized = unnormalized

            # calculate
            avg = np.sum(unnormalized, axis=0) / unnormalized.shape[0]
            dsp = np.sum(np.power((unnormalized - avg), 2), axis=0) / unnormalized.shape[0]
            xi = (unnormalized - avg) / np.sqrt(dsp + e)
            normalized = xi * self.r + self.b

            # update
            self.batch_avg = avg
            self.batch_dsp = dsp
            self.batch_normalized_xi = xi
            self.batch_avg_sum += avg
            self.batch_dsp_sum += dsp
            self.counter += 1


        else:
            # calculate
            dsp_avg = self.batch_dsp_sum / self.counter
            avg_avg = self.batch_avg_sum / self.counter
            std_avg = np.sqrt(dsp_avg + e)
            normalized = self.r * (unnormalized - avg_avg) / std_avg + self.b

        return normalized

    def backward(self, grad_normalized):
        # load
        r = self.r
        xi = self.unnormalized
        _xi = self.batch_normalized_xi
        avg = self.batch_avg
        dsp = self.batch_dsp

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
        self.r += self.optimizer_r.pop(deltaR)
        self.b += self.optimizer_b.pop(deltaB)

        return grad_unnormalized

class Reshaper:

    def __init__(self, channel_num, input_x, input_y):
        self.cn = channel_num
        self.ix = input_x
        self.iy = input_y

    def forward(self, x, it):
        return x.reshape(x.shape[0], self.cn * self.ix * self.iy)

    def backward(self, y):
        return y.reshape(y.shape[0], self.cn , self.ix, self.iy)

class Layer:

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _relu(self, x):
        return np.where(x > 0, x, 0.0)

    def _dropout(self, x, is_training):
        if is_training:
            return np.where(x > 0, x * self.dropouter, 0.0)

        else:
            return np.where(x > 0, x * (1 - dropout_rate), 0.0)

    def _softmax(self, x):
        c = np.max(x, axis=1)  # todo
        exp_x = np.exp((x.T - c).T)
        return (exp_x.T / np.sum(exp_x, axis=1)).T  # todo

    def _drop(self, shape):
        dropouter = np.ones(shape)
        node_drop = int(shape[1] * dropout_rate)
        for i in range(shape[0]):
            dropouter[i][np.random.randint(0, shape[1] - 1, node_drop)] = 0
        return dropouter

    def _activation(self, y, is_training):
        if   self.fun == 'sigmoid': return self._sigmoid(y)
        elif self.fun == 'relu':    return self._relu(y)
        elif self.fun == 'softmax': return self._softmax(y)
        elif self.fun == 'dropout':
            self.dropouter = self._drop(y.shape)
            return self._dropout(y, is_training)

    def _back_activation(self, grad_z):
        if   self.fun == 'sigmoid': return grad_z * (1 - grad_z)
        elif self.fun == 'relu':    return np.where(self.y > 0, grad_z, 0.0)
        elif self.fun == 'softmax': return grad_z # todo grad_y = grad_z
        elif self.fun == 'dropout':
            return np.where(self.y > 0, self.dropouter * grad_z, 0.0)

    def forward(self, data, it):
        return data

    def backward(self, grad):
        return grad

class DenceLayer(Layer):

    def __init__(self, input, output, fun, method):

        self.w = np.random.normal(0, 1 / math.sqrt(input), (input, output))
        self.b = np.random.normal(0, 1 / math.sqrt(input), (output,))
        self.fun = fun
        self.optimizer_w = Optimizer(method)
        self.optimizer_b = Optimizer('Adam')

    def forward(self, data, is_training):

        self.x = data
        self.y = np.dot(self.x, self.w) + self.b
        self.z = self._activation(self.y, is_training)
        return self.z

    def backward(self, grad_z):
        grad_y = self._back_activation(grad_z)
        grad_x = np.dot(grad_y, self.w.T)
        self.w += self.optimizer_w.pop(np.dot(self.x.T, grad_y))
        self.b += self.optimizer_b.pop(grad_y.sum(axis=0))
        return grad_x

class ConvolutionLayer(Layer):

    def __init__(self, channnel_num, filter_num, filter_size, fun, method):
        self.cn = channnel_num
        self.fn = filter_num
        self.fs = filter_size
        self.fun = fun
        self.w = np.random.normal(0, 1 / math.sqrt(10), (self.fn, self.fs * self.fs * self.cn))
        self.b = np.random.normal(0, 1 / math.sqrt(10), (self.fn,))
        self.optimizer_w = Optimizer('Adam')
        self.optimizer_b = Optimizer('Adam')

    # convolution
    # 入力画像と出力画像のサイズは統一想定
    # todo if input_x != input_y...
    # todo @param : stride
    # @param x      :(batch_size, channel_num, input_x, input_y)
    # @param fn     :filter_num (=>出力画像のチャンネル数)
    # @param fn     :filter_size
    # @return       :(batch_size, fn, input_x, input_y)
    def forward(self, x, it):
        self.ps = int(self.fs / 2)
        self.bs, self.cn, self.ix, self.iy = x.shape

        padded = np.pad(x, [(0, 0), (0, 0), (self.ps, self.ps), (self.ps, self.ps)], 'constant').transpose(0, 2, 3, 1)

        # as_strided によって padded から 以下の感じで参照を切り出して整列
        # [0   0   0]       [[10792 10793     0]
        # [0   0   1]        [10798 10799     0]
        # [0  28  29]] ...   [    0     0     0]]]
        dim = padded.shape
        submatrices = as_strided(padded, (self.bs, self.ix, self.iy, self.cn, self.fs, self.fs),
                                 (dim[1] * dim[2] * 8, dim[1] * 8, 8, dim[0] * dim[1] * dim[2] * 8, dim[2] * 8, 8))

        self.x = submatrices.reshape(self.bs * self.ix * self.iy, self.cn * self.fs * self.fs).T

        # test
        # print(np.min(x), np.max(x))
        # print(x.T[0].reshape(cn,fs,fs))
        # print(x.T[-1].reshape(cn,fs,fs))

        # 畳み込む
        # (7840,3) + (3,)
        self.y = np.dot(self.w, self.x).T + self.b
        # print(self.y.reshape(self.bs, self.ix, self.iy, self.fn).shape)

        self.y = self.y.reshape(self.bs, self.ix, self.iy, self.fn).transpose(0, 3, 1, 2)

        self.z = self._activation(self.y, it)


        return self.z

    def backward(self, grad_z):

        grad_y = self._back_activation(grad_z)


        # print(self.w.T.shape, grad_y.transpose(1,0,2,3,).reshape(self.fn, self.bs * self.ix * self.iy).shape, )
        grad_y = grad_y.transpose(1,0,2,3,).reshape(self.fn, self.bs * self.ix * self.iy).T

        grad_x = np.dot(grad_y, self.w)
        grad_w = np.dot(grad_y.T, self.x.T)
        grad_b = grad_y.sum(axis=0)
        self.w += self.optimizer_w.pop(grad_w)
        self.b += self.optimizer_b.pop(grad_b)
        # print(grad_x.shape)

        return grad_x.reshape(self.bs*self.ix*self.iy*self.cn, self.fs * self.fs).sum(axis=1).reshape(self.bs, self.cn, self.ix, self.iy)

class PoolingLayer(Layer):

    def __init__(self, pooling_size):
        self.ps = pooling_size

    # pooling
    # convolution層ではfilter_num個のchannelを持つ画像を生成していると捉えられるので
    # pooling層でのchannel_numは直前のconvolution層でのfilter_numに相当する
    # 左上から切り出すので、input_x/ps, input_y/psの端数分だけ右端下端が切り落とされる
    # @param x      :(batch_size, channel_num, input_x, input_y)
    # @param ps     :pooling_size
    # @return       :(batch_size, channel_num, lx, ly)
    def forward(self, x, it):

        dim = x.shape
        self.bs, self.cn, self.lx, self.ly= dim[0], dim[1], int(dim[2] / self.ps), int(dim[3] / self.ps)

        submatrices = as_strided(x, (self.bs, self.cn, self.lx, self.ly, self.ps, self.ps), (
            dim[1] * dim[2] * dim[3] * 8, dim[2] * dim[3] * 8, dim[2] * 8 * self.ps, 8 * self.ps, dim[3] * 8, 8))

        # max pooling
        a = np.zeros_like(submatrices)
        a = np.where(True, submatrices,0)
        b = a.reshape(self.bs * self.cn * self.lx * self.ly, self.ps * self.ps)
        y = np.max(b, axis=-1)
        self.address = np.identity(self.ps * self.ps)[np.argmax(b, axis=1)].reshape(self.bs,self.cn,self.lx,self.ly,self.ps,self.ps).transpose(0,1,2,4,3,5).reshape(self.bs,self.cn,self.lx*self.ps,self.ly*self.ps)
        # print(self.address)

        # test
        test = y.reshape((self.bs, self.cn * self.lx * self.ly))
        # print(test[0].reshape(self.cn, self.lx, self.ly))
        # print(test[-1].reshape(self.cn, self.lx, self.ly))

        return y.reshape(self.bs, self.cn, self.lx, self.ly)

    def backward(self, grad_y):
        padded = np.pad(grad_y.reshape(self.bs,self.cn,self.lx,self.ly,1,1), [(0,0),(0,0),(0,0),(0,0),(self.ps-1,0),(self.ps-1,0)], 'edge').transpose(0,1,2,4,3,5).reshape(self.bs,self.cn,self.lx*self.ps,self.ly*self.ps)
        # print(padded)
        a =padded * self.address
        # print(a[0][0])
        return a

class Network:

    def __init__(self, network_setting):
        self.ns = network_setting
        self.len = len(self.ns)

    def _forward_propagation(self, input, it):

        x = input
        for i in range(self.len):
            x = self.ns[i].forward(x, it)
        return x

    def _back_propagation(self, output, label):

        x = (output - label) / batch_size
        for i in range(self.len):
            x = self.ns[self.len - i - 1].backward(x)
        return x

    def _cross_entropy_error(self, x, y):
        return -np.sum(y * np.log(x)) / x.shape[0]

    def _chose_batch(self, X, Y):
        batch = np.random.randint(0, X.shape[0] - 1, batch_size)
        return X[batch], Y[batch]

    def training(self, X, Y):
        input, label = self._chose_batch(X, Y)
        output = self._forward_propagation(input, True)
        # print(output.argmax(axis=1), Y.argmax(axis=1))
        a = np.sum( output.argmax(axis=1) == label.argmax(axis=1) )  / input.shape[0]
        # print(a)
        self._back_propagation(output, label)
        return self._cross_entropy_error(output, label)

    def testing(self, X, Y):
        input, label = self._chose_batch(X, Y)
        return np.sum(self._forward_propagation(input, False).argmax(axis=1) == label.argmax(axis=1)) / input.shape[0]

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

### set constants ###
np.random.seed(0)
dataset = 'mnist'
optimization_methods = ['Adam']
# optimization_methods = ['SGD', 'Adam']
# optimization_methods = ['SGD', 'MSGD', 'AdaGrad', 'RMSProp', 'AdaDelta', 'Adam']
batch_size = 100  # 100
epoch = 3  # 10
e = 1e-7
dropout_rate = 0.25


### load dataset ###
print('loading data ... ', end="")
loader = DataLoader(dataset)
trainX, trainY = loader.load_train_data()
testX, testY = loader.load_test_data()
print('done')

### initialize result boxs ###
times_per_epoch = loader.train_data_size / batch_size
times = int(epoch * times_per_epoch)
result = {}
result['time'] = {}
result['test'] = {}
result['train'] = {}
for method in optimization_methods:
    result['train'][method] = np.zeros(times)
    result['test'][method] = np.zeros(epoch)


for method in optimization_methods:

    print()
    print('learning by {0}'.format(method))

    ### init network ###

    # How to set network setting
    # ConvolutionLayer(channnel_num, filter_num, filter_size, fun, method),
    # Reshaper(channel_num, input_x, input_y),
    # DenceLayer(input, output(=node_size), fun, method),

    pooling_size = 1
    network = Network({
        # input (batch_size,1,28,28)
        0: Layer(),
        # # 0: ConvolutionLayer(1, 2, 3,'dropout', method),
        1: Layer(),
        # 1: ConvolutionLayer(1, 3, 3, 'dropout', method),
        2: Layer(),
        # 2: PoolingLayer(pooling_size),
        3: Reshaper(1, int(28 / pooling_size), int(28 / pooling_size)),
        4: DenceLayer(1 * int(28 / pooling_size) * int(28 / pooling_size), 600, 'dropout', method),
        5: BatchNormalizer(),
        6: DenceLayer(600, 600, 'dropout', method),
        7: BatchNormalizer(),
        8: DenceLayer(600, 10, 'softmax', method),

        # 0: ConvolutionLayer(1, 3, 5, 'dropout', method),
        # # 1: ConvolutionLayer(3, 3, 5, 'relu', method),
        # 1: PoolingLayer(2),
        # 2: Reshaper(3, 14, 14),
        # 3: DenceLayer(3 * 14 * 14, 10, 'softmax', method),
    })

    start = time.time()

    for i in range(result['train'][method].size):

        ### training per batch ###
        result['train'][method][i] = network.training(trainX, trainY)
        # print(result['train'][method][i])

        ### testing per epoch ###
        if i % times_per_epoch == 0:
            index = int(i / times_per_epoch)
            result['test'][method][index] = network.testing(testX, testY)
            print('epoch {0} train error : {1}  test error : {2}'.format(index, result['train'][method][i], result['test'][method][index]))

    result['time'][method] = time.time() - start

# print result
print()
print('!!!result!!!')
print()
for method in optimization_methods:
    print(method)
    print('time : {0}sec'.format(result['time'][method]))
    print(result['test'][method])
    print('accuracity average : {0}'.format(np.average(result['test'][method])))
    print()

    plot(result['train'][method], method)

plt.show()
