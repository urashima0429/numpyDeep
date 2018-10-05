import math
import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from pylab import cm
mndata = MNIST("/export/home/016/a0165336/project/le4nn/")

# constants
def init():
    m = 10
    np.random.seed(0)
    network = {}
    network['w1'] = np.random.normal(0, 1/math.sqrt(784),(784,m))
    network['b1'] = np.random.normal(0, 1/math.sqrt(784),(m, ))
    network['w2'] = np.random.normal(0, 1/math.sqrt(m),(m,10))
    network['b2'] = np.random.normal(0, 1/math.sqrt(m),(10, ))
    return network

def sigmoid(x):
    return np.where( x < -709, 0.0, 1 / (1 + np.exp(-x)))

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    return exp_x / np.sum(exp_x)

def cross_entropy_error(x, y):
    return np.sum(-y * np.log(x), axis=1)

# 前処理 (10000, 28, 28) => (28, 28)
# network = init()
# print("loading test data...")
# X, Y = mndata.load_testing()
# X = np.array(X)
# X = X.reshape((X.shape[0],28,28))
# Y = np.array(Y)
# print("input num 0~9999 >>>")
# idx = int(input())

# # 入力層 (28,28) => (784, )
# x = X[idx].reshape( (784, ) )
#
# # 中間層への入力を計算する層（全結合層） (m, 784)(784, ) => (m, )
# x = np.dot(network['w1'], x) + network['b1']
#
# # 中間層 シグモイド関数 (m, ) => (m, )
# x = sigmoid(x)
#
# # 出力層への入力を計算する層（全結合層） (10, m)(m, ) => (10, )
# x = np.dot(network['w2'], x) + network['b2']
#
# # 出力層 ソフトマックス関数 (10, ) => (10, )
# x = softmax(x)
#
# # 後処理
# print(x)
# print(np.where(x == np.max(x)))
#
# print(cross_entropy_error(x, np.identity(10)[Y[idx]]))

# plt.imshow(X[idx], cmap=cm.gray)
# plt.show()



# 前処理 (10000, 28, 28) => (28, 28)
network = init()
print("loading test data...")
X, Y = mndata.load_training()
X = np.array(X)
X = X.reshape((X.shape[0],28,28))
Y = np.array(Y)

b_size = 100
batch = np.random.randint(0,59999,b_size)

# 入力層 (b_size, 28,28) => (b_size, 784)
x = X[batch].reshape( (b_size, 784) )

# 中間層への入力を計算する層（全結合層） (b_size, 784)(784, m) => (b_size, m)
x = np.dot(x, network['w1']) + network['b1']

# 中間層 シグモイド関数 (b_size, m) => (b_size, m)
x = sigmoid(x)

# 出力層への入力を計算する層（全結合層） (b_size, m)(m, 10) => (b_size, 10)
x = np.dot(x, network['w2']) + network['b2']

# 出力層 ソフトマックス関数 (b_size, 10) => (b_size, 10)
x = softmax(x)

# 後処理
x = cross_entropy_error(x, np.identity(10)[Y[batch]])
print(np.average(x))