import numpy as np

a = np.array([1,3,4])
b = np.array([7, 8, 9])
c = np.array([10, 11, 12])
print(np.concatenate((a, b, c), axis=1))
print(np.concatenate((a, b.T), axis=1))
print(np.concatenate((a, b), axis=None))
