import numpy as np

record = float('inf')
path = []


def splice_index_vector(x_vec, y_vec, index_x, index_y):
    x_vec = np.concatenate((x_vec[:index_x], x_vec[index_x + 1:]))
    y_vec = np.concatenate((y_vec[:index_y], y_vec[index_y + 1:]))
    return x_vec, y_vec


def get_cost_coefficient(row, column, index):
    i, j = index[0], index[1]
    s_row = np.concatenate((row[:j], row[j + 1:]))
    s_col = np.concatenate((column[:i], column[i + 1:]))
    return min(s_row) + min(s_col)


def check_pzc(matrix):
    n = matrix[0].size
    for i in np.arange(0, n, 1):
        if float('inf') not in matrix[i]:
            for j in np.arange(0, n, 1):
                if float('inf') not in matrix[:, j]:
                    matrix[i][j] = float('inf')
                    break


def solve(cost_matrix, way, limit, x_vec, y_vec):
    n = cost_matrix[0].size
    global record, path
    if n > 2:
        min_row_val = cost_matrix.min(axis=1)
        if float('inf') in min_row_val:
            return
        for i in np.arange(0, n, 1):
            cost_matrix[i] -= min_row_val[i]
        min_col_val = cost_matrix.min(axis=0)
        if float('inf') in min_col_val:
            return
        limit += sum(min_col_val) + sum(min_row_val)
        if record < limit:
            return
        for i in np.arange(0, n, 1):
            if min_col_val[i] != 0:
                cost_matrix[:, i] -= min_col_val[i]
        zeroes = []
        for i in np.arange(0, n, 1):
            for j in np.arange(0, n, 1):
                if cost_matrix[i, j] == 0:
                    cost_coefficient = get_cost_coefficient(cost_matrix[i, :], cost_matrix[:, j], (i, j))
                    zeroes.append((i, j, cost_coefficient))
        zeroes.sort(reverse=True, key=lambda x: x[2])
        x, y = zeroes[0][0], zeroes[0][1]
        const_matrix_copy = np.empty_like(cost_matrix)
        const_matrix_copy[:] = cost_matrix
        const_matrix_copy[x][y] = float('inf')
        cost_matrix = np.delete(cost_matrix, (zeroes[0][0]), axis=0)
        cost_matrix = np.delete(cost_matrix, (zeroes[0][1]), axis=1)
        way_copy = way[:]
        way.append((x_vec[x], y_vec[y]))
        x_vec_copy, y_vec_copy = splice_index_vector(x_vec, y_vec, zeroes[0][0], zeroes[0][1])
        check_pzc(cost_matrix)
        solve(cost_matrix, way, limit, x_vec_copy, y_vec_copy)
        solve(const_matrix_copy, way_copy, limit, x_vec, y_vec)
    if n == 2:
        for i in np.arange(0, 2, 1):
            if cost_matrix[i][0] != float('inf'):
                way.append((x_vec[i], y_vec[0]))
                limit += cost_matrix[i][0]
            else:
                way.append((x_vec[i], y_vec[1]))
                limit += cost_matrix[i][1]
        if record > limit:
            record = limit
            path = way[:]
    return way, limit


def little(cost_matrix):
    way = []
    n = cost_matrix[0].size
    x_vec = np.arange(0, n)
    y_vec = np.arange(0, n)
    solve(cost_matrix, way, 0, x_vec, y_vec)
    global record, path
    return path
