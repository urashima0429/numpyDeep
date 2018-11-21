import numpy as np


a = np.arange(3 * 3 * 7).reshape(3 * 3, 7)
b = np.max(a, axis=1)
c = np.identity(7)[np.argmax(a, axis=1)]
print(a, b, c)
# z = np.where(submatrices == y, 1, 0)