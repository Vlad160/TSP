import numpy as np


def solve(stretch_matrix, start):
    way = []
    n = stretch_matrix[0].size
    way.append(start)
    for i in np.arange(1, n, 1):
        s = stretch_matrix[way[i - 1]].tolist()
        way.append(s.index(min(s)))
        for j in np.arange(0, i, 1):
            stretch_matrix[way[i], way[j]] = float('inf')
            stretch_matrix[way[j], way[i]] = float('inf')
    return way
