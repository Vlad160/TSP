import numpy as np
import time
from src.algorithms import nearest_neighbor, prim_euler, tsp_genetic_python, ant
from src.generators import pretty_gen
from src.plotters import plotter
import matplotlib.pyplot as plt


def nearest_neighbour(start, matrix, cities_list):
    m_copy = np.empty_like(matrix)
    m_copy[:] = matrix
    t1 = time.time()
    way = nearest_neighbor.solve(m_copy, start)
    t2 = time.time() - t1
    print(t2)
    cities_way = [cities_list[way[i]] for i in np.arange(0, len(cities_list), 1)]
    cities_way.append(cities_way[0])
    s = get_sum(cities_way)
    plotter.plot_cities(cities_way, s)
    return way, s


def nearest_neighbour_modification(matrix, cities_list):
    min_way = []
    min_sum = []
    n = matrix[0].size
    t1 = time.time()
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
    t2 = time.time() - t1
    print(t2)
    plotter.plot_cities(min_way[min_sum_index], min_sum[min_sum_index])
    return min_way[min_sum_index], min_sum[min_sum_index]


def prim_euler_solve(matrix, cities_list):
    m_copy = np.empty_like(matrix)
    m_copy[:] = matrix
    t1 = time.time()
    way = prim_euler.solve(m_copy)
    t2 = time.time() - t1
    print(t2)
    cities_way = [cities_list[way[i]] for i in np.arange(0, len(cities_list), 1)]
    cities_way.append(cities_way[0])
    s = get_sum(cities_way)
    plotter.plot_cities(cities_way, s)
    return way, s


def genetic(cities_list):
    t1 = time.time()
    cities_way = tsp_genetic_python.random_cities(cities_list)
    cities_way.append(cities_way[0])
    t2 = time.time() - t1
    print(t2)
    s = get_sum(cities_way)
    plotter.plot_cities(cities_way, s)
    return cities_way, s


def aio(matrix, cities_list):
    m_copy = np.empty_like(matrix)
    m_copy[:] = matrix
    t1 = time.time()
    cities_way = ant.solve(cities_list, matrix)
    t2 = time.time() - t1
    print(t2)
    cities_way.append(cities_way[0])
    s = get_sum(cities_way)
    plotter.plot_cities(cities_way, s)
    return cities_way, s


def get_sum(way):
    length = 0.0
    for city in way:
        next_city = way[way.index(city) - len(way) + 1]
        dist_to_next = city.distance_to[next_city.name]
        length += dist_to_next
    return length


def compare_algorithms():
    cities_count_list = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    colors = ['r', 'b', 'g', 'y', 'k']
    labels = ['Ближний сосед', 'Ближний сосед(из всех)', 'Прима-Эйлера', 'Генетический', 'Муравьиный']
    max_length = 500
    comp_s = []
    comp_len = []
    handlers = []
    for _ in np.arange(0, 5, 1):
        comp_s.append(0)
        comp_len.append(0)
    for cities_count in cities_count_list:
        cities, distance_matrix = pretty_gen.generate(cities_count, max_length)
        way, s = nearest_neighbour(0, distance_matrix, cities)
        comp_s.append(s)
        comp_len.append(cities_count)
        way, s = nearest_neighbour_modification(distance_matrix, cities)
        comp_s.append(s)
        comp_len.append(cities_count)
        way, s = prim_euler_solve(distance_matrix, cities)
        comp_s.append(s)
        comp_len.append(cities_count)
        way, s = genetic(cities)
        comp_s.append(s)
        comp_len.append(cities_count)
        way, s = aio(distance_matrix, cities)
        comp_s.append(s)
        comp_len.append(cities_count)
    for i in np.arange(0, 5, 1):
        handle, = plt.plot(comp_len[i::5], comp_s[i::5], color=colors[i], linewidth='2', label=labels[i])
        handlers.append(handle)
    plt.legend(handles=handlers, loc=3)
    plt.grid(True)
    plt.show()


def compare_nearest_neighbour():
    cities_count_list = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    colors = ['r', 'b']
    labels = ['Ближний сосед', 'Ближний сосед(из всех)']
    max_length = 500
    comp_s = []
    comp_len = []
    handlers = []
    for _ in np.arange(0, 2, 1):
        comp_s.append(0)
        comp_len.append(0)
    for cities_count in cities_count_list:
        way_len = 0
        cities, distance_matrix = pretty_gen.generate(cities_count, max_length)
        for i in np.arange(20):
            way, s = nearest_neighbour(0, distance_matrix, cities)
            way_len += s
        comp_s.append(way_len / 20)
        comp_len.append(cities_count)
        way_len = 0
        for i in np.arange(20):
            way, s = nearest_neighbour_modification(distance_matrix, cities)
            way_len += s
        comp_s.append(way_len / 20)
        comp_len.append(cities_count)
    for i in np.arange(0, 2, 1):
        handle, = plt.plot(comp_len[i::2], comp_s[i::2], color=colors[i], linewidth='2', label=labels[i])
        handlers.append(handle)
    plt.legend(handles=handlers, loc=3)
    plt.grid(True)
    plt.show()


cities, distance_matrix = pretty_gen.generate(25, 500)
nearest_neighbour(0, distance_matrix, cities)
nearest_neighbour_modification(distance_matrix, cities)
prim_euler_solve(distance_matrix, cities)
genetic(cities)
aio(distance_matrix, cities)
# little_solve(distance_matrix, cities)
# compare_nearest_neighbour()
# compare_algorithms()
