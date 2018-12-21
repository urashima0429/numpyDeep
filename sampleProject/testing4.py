import time
import math
import numpy as np
from mnist import MNIST
import pickle
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided

dropout_rate = 0.40

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
    # self.x.shape  :(batch_size * input_x * input_y, channel_num * filter_size * filter_size)
    # self.y.shape  :(batch_size, self.fn * self.ix * self.iy)
    # self.z.shape  :(batch_size, filter_num,  input_x, input_y)
    # @return       :(batch_size, filter_num,  input_x, input_y)
    def forward(self, x, it):
        self.ps = int(self.fs / 2)
        self.bs, self.cn, self.ix, self.iy = x.shape

        # 0-padding
        padded = np.pad(x, [(0, 0), (0, 0), (self.ps, self.ps), (self.ps, self.ps)], 'constant').transpose(0, 2, 3, 1)

        # as_strided によって padded から 以下の感じで参照を切り出して整列
        # [0   0   0]       [[10792 10793     0]
        # [0   0   1]        [10798 10799     0]
        # [0  28  29]] ...   [    0     0     0]]]
        dim = padded.shape
        submatrices = as_strided(padded, (self.bs, self.ix, self.iy, self.cn, self.fs, self.fs),
                                 (dim[1] * dim[2] * 8, dim[1] * 8, 8, dim[0] * dim[1] * dim[2] * 8, dim[2] * 8, 8))

        # make up large x
        self.x = submatrices.reshape(self.bs * self.ix * self.iy, self.cn * self.fs * self.fs).T

        # convolution
        self.y = (np.dot(self.w, self.x).T + self.b).reshape(self.bs, self.ix, self.iy, self.fn).transpose(0, 3, 1, 2).reshape(self.bs, self.fn * self.ix * self.iy)

        self.z = self._activation(self.y, it).reshape(self.bs, self.fn, self.ix, self.iy)

        return self.z

    def backward(self, grad_z):

        grad_y = self._back_activation(grad_z.reshape(self.bs, self.fn * self.ix * self.iy)).reshape(self.bs, self.fn, self.ix, self.iy)

        grad_y = grad_y.transpose(1,0,2,3,).reshape(self.fn, self.bs * self.ix * self.iy).T

        grad_x = np.dot(grad_y, self.w) #todo
        grad_w = np.dot(grad_y.T, self.x.T)
        grad_b = grad_y.sum(axis=0)
        self.w += self.optimizer_w.pop(grad_w)
        self.b += self.optimizer_b.pop(grad_b)


        return

class PoolingLayer(Layer):

    def __init__(self, pooling_size):
        self.ps = pooling_size

    # pooling
    # convolution層ではfilter_num個のchannelを持つ画像を生成していると捉えられるので
    # pooling層でのchannel_numは直前のconvolution層でのfilter_numに相当する
    # 左上から切り出すので、input_x/ps, input_y/psの端数分だけ右端下端が切り落とされる
    # @param x      :(batch_size, channel_num, input_x, input_y)
    # @return       :(batch_size, channel_num, lx, ly)
    def forward(self, x, it):

        dim = x.shape
        self.bs, self.cn, self.lx, self.ly= dim[0], dim[1], int(dim[2] / self.ps), int(dim[3] / self.ps)

        submatrices = as_strided(x, (self.bs, self.cn, self.lx, self.ly, self.ps, self.ps), (
            dim[1] * dim[2] * dim[3] * 8, dim[2] * dim[3] * 8, dim[2] * 8 * self.ps, 8 * self.ps, dim[3] * 8, 8))

        # max pooling
        submatrices = submatrices.reshape(self.bs * self.cn * self.lx * self.ly, self.ps * self.ps)
        y = np.max(submatrices, axis=-1)
        self.address = np.identity(self.ps * self.ps)[np.argmax(submatrices, axis=1)].reshape(self.bs,self.cn,self.lx,self.ly,self.ps,self.ps).transpose(0,1,2,4,3,5).reshape(self.bs,self.cn,self.lx*self.ps,self.ly*self.ps)

        return y.reshape(self.bs, self.cn, self.lx, self.ly)

    def backward(self, grad_y):
        padded = np.pad(grad_y.reshape(self.bs,self.cn,self.lx,self.ly,1,1), [(0,0),(0,0),(0,0),(0,0),(self.ps-1,0),(self.ps-1,0)], 'edge').transpose(0,1,2,4,3,5).reshape(self.bs,self.cn,self.lx*self.ps,self.ly*self.ps)
        return padded * self.address

# a = np.arange(2*3*6*6).reshape((2,3,6,6))
a = np.random.normal(0, 1 / math.sqrt(10), (2,3,3,3))

conv = ConvolutionLayer(3, 3, 3, 'relu', 'Adam')

b = conv.forward(a, 0)