import numpy as np

test = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
test2 = np.copy(test)
test2[0, 0] = 100
print(test)