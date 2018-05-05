import numpy as np

path = []


def prim(stretch_matrix):
    n = stretch_matrix[0].size
    mst_edges = np.empty_like(stretch_matrix)
    mst_edges.fill(0)
    used = np.zeros(n, dtype=bool)
    min_e = np.zeros(n)
    min_e.fill(np.math.inf)
    sel_e = np.zeros(n, dtype=int)
    sel_e.fill(-1)
    min_e[0] = 0
    for _ in np.arange(0, n, 1):
        v = -1
        for j in np.arange(0, n, 1):
            if not used[j] and (v == -1 or min_e[j] < min_e[v]):
                v = j
        if min_e[v] == np.math.inf:
            print('NO MST!')
            return
        used[v] = True
        if sel_e[v] != -1:
            mst_edges[sel_e[v], v] = 1
            mst_edges[v, sel_e[v]] = 1
        for to in np.arange(0, n, 1):
            if not used[to] and (stretch_matrix[v, to] < min_e[to]):
                min_e[to] = stretch_matrix[v, to]
                sel_e[to] = v
    return mst_edges


def find_path(mst_matrix, v):
    global path
    path.append(v)
    n = mst_matrix[0].size
    for i in np.arange(0, n, 1):
        if mst_matrix[v, i] == 1:
            mst_matrix[v, i] = 0
            mst_matrix[i, v] = 0
            find_path(mst_matrix, i)


def solve(stretch_matrix):
    global path
    path = []
    n = stretch_matrix[0].size
    mst_tree = prim(stretch_matrix)
    v = 0
    for i in np.arange(0, n, 1):
        if np.sum(mst_tree[i]) == 1:
            v = i
            break
    find_path(mst_tree, v)
    return path
