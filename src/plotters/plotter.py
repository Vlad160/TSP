import matplotlib.pyplot as plt
import numpy as np


def plot(way_length, x, y, way, ib, n, m):
    plt.title('Общий путь-%s. Номер города-%i. Всего городов -%i.\n Координаты X,Y случайные числа от %i до %i' % (
        round(way_length, 3), ib, n, 0, m), size=14)
    x1 = [x[way[i]] for i in np.arange(0, n, 1)]
    y1 = [y[way[i]] for i in np.arange(0, n, 1)]
    plt.plot(x1, y1, color='r', linestyle=' ', marker='o')
    plt.plot(x1, y1, color='b', linewidth=1)
    x2 = [x[way[n - 1]], x[way[0]]]
    y2 = [y[way[n - 1]], y[way[0]]]
    line_start_end, = plt.plot(x2, y2, color='g', linewidth=2, linestyle='-',
                               label='Путь от  последнего \n к первому городу')
    end, = plt.plot(x[way[n - 1]], y[way[n - 1]], color='b', linestyle=' ', marker='o', label='Конечный город')
    start, = plt.plot(x[way[0]], y[way[0]], color='y', linestyle=' ', marker='o', label='Начальный город')
    plt.legend(handles=[start, end, line_start_end], loc=3)
    plt.grid(True)
    plt.show()


def plot_edges(edges, x, y):
    n = len(edges)
    x1 = [x[edges[i][0]] for i in np.arange(0, n, 1)]
    y1 = [y[edges[i][1]] for i in np.arange(0, n, 1)]
    plt.plot(x1, y1, color='r', linestyle=' ', marker='o')
    plt.plot(x1, y1, color='b', linewidth=1)
    x2 = [x1[n - 1], x1[0]]
    y2 = [y1[n - 1], y1[0]]
    line_start_end, = plt.plot(x2, y2, color='g', linewidth=2, linestyle='-',
                               label='Путь от  последнего \n к первому городу')
    end, = plt.plot(x1[n - 1], y1[n - 1], color='b', linestyle=' ', marker='o', label='Конечный город')
    start, = plt.plot(x1[0], y1[0], color='y', linestyle=' ', marker='o', label='Начальный город')
    plt.grid(True)
    plt.show()


def plot_cities(cities, length):
    n = len(cities)
    plt.title('Общий путь-%s. Всего городов -%i.' % (
        round(length, 3), n), size=14)
    x1 = [cities[i].x for i in np.arange(0, n, 1)]
    y1 = [cities[i].y for i in np.arange(0, n, 1)]
    plt.plot(x1, y1, color='r', linestyle=' ', marker='o')
    plt.plot(x1, y1, color='b', linewidth=1)
    plt.plot(x1[n - 2], y1[n - 2], color='b', linestyle=' ', marker='o', label='Конечный город')
    plt.plot(x1[0], y1[0], color='y', linestyle=' ', marker='o', label='Начальный город')
    plt.grid(True)
    plt.show()
