import math
import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from pylab import cm
mndata = MNIST("/export/home/016/a0165336/project/le4nn/")

m = 10  # node size
np.random.seed(0)

def init():
    network = {}
    network['w1'] = np.random.normal(0, 1/math.sqrt(784),(784,m))
    network['b1'] = np.random.normal(0, 1/math.sqrt(784),(m, ))
    network['w2'] = np.random.normal(0, 1/math.sqrt(m),(m,10))
    network['b2'] = np.random.normal(0, 1/math.sqrt(m),(10, ))
    return network

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    return exp_x / np.sum(exp_x)

# 前処理 (10000, 28, 28) => (28, 28)
network = init()
print("loading test data...")
X, Y = mndata.load_testing()
X = np.array(X)
X = X.reshape((10000,28,28))
Y = np.array(Y)
print("input num 0~9999 >>>")
idx = int(input())

# 入力層 (28,28) => (784, )
x = X[idx].reshape( (784, ) )

# 中間層への入力を計算する層（全結合層） (784, )(784, m) => (m, )
x = np.dot(x, network['w1']) + network['b1']

# 中間層 シグモイド関数 (m, ) => (m, )
x = sigmoid(x)

# 出力層への入力を計算する層（全結合層） (m, )(m, 10) => (10, )
x = np.dot(x, network['w2']) + network['b2']

# 出力層 ソフトマックス関数 (10, ) => (10, )
x = softmax(x)

# 後処理
print(x)
print(x.argmax())

# plt.imshow(X[idx], cmap=cm.gray)
# plt.show()