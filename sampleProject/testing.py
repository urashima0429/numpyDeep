import numpy as np

def sigmoid(x):
    return np.where( x < -709, 0.0, 1 / (1 + np.exp(-x)))

def cross_entropy_error(x, y):
    return -np.sum(y * np.log(x)) / x.shape[0]

a = (np.arange(20) + 1).reshape(2,10)
b = np.array((1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1)).reshape(2,10)
c = cross_entropy_error(a,b)
print(c)
