import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from pylab import cm
mndata = MNIST("/export/home/016/a0165336/project/le4nn/")


X, Y = mndata.load_training()
X = np.array(X)
X = X.reshape((X.shape[0],28,28))
Y = np.array(Y)


print("input num 0~9999")
idx = int(input())

print(Y[idx])
plt.imshow(X[idx], cmap=cm.gray)
plt.show()



