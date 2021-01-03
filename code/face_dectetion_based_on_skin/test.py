import numpy as np

a = {"a":[1, 2], "b":[2, 3]}
b = np.array(list(a.values()))
print(np.array([(b - b[:, 0].reshape(-1, 1))[:, 1] * np.random.uniform(low = 0, high = 1, size = 2) + b[:, 0] for i in range(10)]))