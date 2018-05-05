import numpy as np


def generate(n, m):
    x = np.random.uniform(0, m, n)
    y = np.random.uniform(0, m, n)
    m = np.zeros([n, n])
    for i in np.arange(0, n, 1):
        for j in np.arange(0, n, 1):
            if i != j:
                m[i, j] = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            else:
                m[i, j] = float('inf')
    return m, x, y
