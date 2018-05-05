import numpy as np
import copy
from src.algorithms import nearest_neighbor, little, prim_euler, tsp_genetic_python
from src.generators.generator import generate
from src.generators import pretty_gen
from src.plotters import plotter
import matplotlib.pyplot as plt
from numpy import sqrt


def nearest_neighbour(start, matrix, cities_list):
    m_copy = np.empty_like(matrix)
    m_copy[:] = matrix
    way = nearest_neighbor.solve(m_copy, start)
    cities_way = [cities_list[way[i]] for i in np.arange(0, len(cities_list), 1)]
    cities_way.append(cities_way[0])
    s = get_sum(cities_way)
    plotter.plot_cities(cities_way, s)
    return way


def nearest_neighbour_modification(matrix, cities_list):
    min_way = []
    min_sum = []
    n = matrix[0].size
    for i in np.arange(0, n, 1):
        m_copy = np.empty_like(matrix)
        m_copy[:] = matrix
        way = nearest_neighbor.solve(m_copy, i)
        cities_way = [cities_list[way[i]] for i in np.arange(0, len(cities_list), 1)]
        cities_way.append(cities_way[0])
        min_way.append(cities_way)
        s = get_sum(cities_way)
        min_sum.append(s)
    min_sum_index = min_sum.index(min(min_sum))
    plotter.plot_cities(min_way[min_sum_index], min_sum[min_sum_index])
    return min_way[min_sum_index], min_sum[min_sum_index]


def little_solve(matrix, cities_list):
    m_copy = np.empty_like(matrix)
    m_copy[:] = matrix
    way = little.little(m_copy)
    path = {}
    s = 0
    for i in np.arange(0, len(way), 1):
        s += matrix[way[i][0]][way[i][1]]
        path[way[i][0]] = way[i][1]
    cities_way = []
    start = next(iter(path))
    cities_way.append(cities_list[start])
    to = path[start]
    while to != start:
        cities_way.append(cities_list[to])
        to = path[to]
    cities_way.append(cities_way[0])
    s = get_sum(cities_way)
    plotter.plot_cities(cities_way, s)
    print(s)


def prim_euler_solve(matrix, cities_list):
    m_copy = np.empty_like(matrix)
    m_copy[:] = matrix
    way = prim_euler.solve(m_copy)
    cities_way = [cities_list[way[i]] for i in np.arange(0, len(cities_list), 1)]
    cities_way.append(cities_way[0])
    s = get_sum(cities_way)
    plotter.plot_cities(cities_way, s)
    return way, s


def genetic(cities_list):
    cities_way = tsp_genetic_python.random_cities(cities_list)
    cities_way.append(cities_way[0])
    s = get_sum(cities_way)
    plotter.plot_cities(cities_way, s)


def get_sum(way):
    length = 0.0
    for city in way:
        next_city = way[way.index(city) - len(way) + 1]
        dist_to_next = city.distance_to[next_city.name]
        length += dist_to_next
    return length


def compare_algorithms():
    cities_count_list = [10, 20, 50, 100, 150, 200, 250]
    colors = ['r', 'b', 'g']
    labels = ['Ближний сосед', 'Ближний сосед(из всех)', 'Прима-Эйлера']
    max_length = 250
    comp_s = []
    comp_len = []
    handlers = []
    for _ in np.arange(0, 3, 1):
        comp_s.append(0)
        comp_len.append(0)
    for cities_count in cities_count_list:
        matrix, x, y = generate(cities_count, max_length)
        way, s = nearest_neighbour(0, matrix, x, y)
        comp_s.append(s)
        comp_len.append(cities_count)
        way, s = nearest_neighbour_modification(matrix, x, y)
        comp_s.append(s)
        comp_len.append(cities_count)
        way, s = prim_euler_solve(matrix, x, y)
        comp_s.append(s)
        comp_len.append(cities_count)
    for i in np.arange(0, 3, 1):
        handle, = plt.plot(comp_len[i::3], comp_s[i::3], color=colors[i], linewidth='2', label=labels[i])
        handlers.append(handle)
    plt.legend(handles=handlers, loc=3)
    plt.grid(True)
    plt.show()


def compare_nearest_neighbour():
    cities_count_list = [10, 20, 50, 100, 150, 200, 250]
    colors = ['r', 'b']
    labels = ['Ближний сосед', 'Ближний сосед(из всех)']
    max_length = 250
    comp_s = []
    comp_len = []
    handlers = []
    for _ in np.arange(0, 2, 1):
        comp_s.append(0)
        comp_len.append(0)
    for cities_count in cities_count_list:
        matrix, x, y = generate(cities_count, max_length)
        way, s = nearest_neighbour(0, matrix, x, y)
        comp_s.append(s)
        comp_len.append(cities_count)
        way, s = nearest_neighbour_modification(matrix, x, y)
        comp_s.append(s)
        comp_len.append(cities_count)
    for i in np.arange(0, 2, 1):
        handle, = plt.plot(comp_len[i::2], comp_s[i::2], color=colors[i], linewidth='2', label=labels[i])
        handlers.append(handle)
    plt.legend(handles=handlers, loc=3)
    plt.grid(True)
    plt.show()


cities, distance_matrix = pretty_gen.generate(10, 150)
# nearest_neighbour(0, distance_matrix, cities)
# prim_euler_solve(distance_matrix, cities)
# nearest_neighbour_modification(distance_matrix, cities)
genetic(cities)
# little_solve(distance_matrix, cities)
# compare_nearest_neighbour()
# compare_algorithms()
