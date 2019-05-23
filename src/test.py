import time
import matplotlib.pyplot as plt
import numpy as np

from src.algorithms import nearest_neighbor, tsp_genetic_python, prim_euler, ant
from src.algorithms.anneal import SimAnneal
from src.generators.pretty_gen import generate, from_file, write_to_file

# cities, matrix = generate(50, 1000)

# write_to_file(cities)
from src.plotters.plotter import plot_cities

coords, distance_matrix = from_file('out.txt')


def calc_fitness(path):
    fitness = 0.0
    for city in path:
        next_city = path[path.index(city) - len(path) + 1]
        dist_to_next = city.distance_to[next_city.name]
        fitness += dist_to_next
    return fitness


def anneal(cities, filename=None):
    sa = SimAnneal(cities, stopping_iter=5000)
    t1 = time.time()
    print('Anneal start at {}'.format(t1))
    sa.anneal()
    t2 = time.time() - t1
    print('Anneal ended after {}'.format(t2))
    solution = sa.best_solution[:]
    solution.append(solution[0])
    path = []
    for i in solution:
        path.append(cities[i])
    fitness = calc_fitness(path)
    print('Anneal fitness {}'.format(fitness))
    if filename:
        sa.visualize_routes(filename)
        sa.plot_learning('learning-anneal.png')
    return path, fitness


def greedy(distance_matrix, cities_list, filename=None):
    m_copy = np.empty_like(distance_matrix)
    m_copy[:] = distance_matrix
    t1 = time.time()
    print('Greedy start at {}'.format(t1))
    path = nearest_neighbor.solve(m_copy, 0)
    t2 = time.time() - t1
    print('Greedy ended after {}'.format(t2))
    path = [cities_list[path[i]] for i in np.arange(0, len(cities_list), 1)]
    path.append(path[0])
    fitness = calc_fitness(path)
    print('Greedy fitness {}'.format(fitness))
    if filename:
        plot_cities(path, fitness, filename)
    return path, fitness


def genetic(cities_list, filename=None):
    t1 = time.time()
    print('Genetic start at {}'.format(t1))
    path = tsp_genetic_python.random_cities(cities_list)
    path.append(path[0])
    t2 = time.time() - t1
    print('Genetic ended after {}'.format(t2))
    fitness = calc_fitness(path)
    print('Genetic fitness {}'.format(fitness))
    if filename:
        plot_cities(path, fitness, filename)
    return path, fitness


def greedy_improved(distance_matrix, cities_list, filename=None):
    min_path = []
    min_fitness = []
    n = distance_matrix[0].size
    t1 = time.time()
    print('Greedy improved start at {}'.format(t1))
    for i in np.arange(0, n, 1):
        m_copy = np.empty_like(distance_matrix)
        m_copy[:] = distance_matrix
        way = nearest_neighbor.solve(m_copy, i)
        cities_way = [cities_list[way[i]] for i in np.arange(0, len(cities_list), 1)]
        cities_way.append(cities_way[0])
        min_path.append(cities_way)
        fitness = calc_fitness(cities_way)
        min_fitness.append(fitness)
    min_sum_index = min_fitness.index(min(min_fitness))
    t2 = time.time() - t1
    print('Greedy improved ended after {}'.format(t2))
    print('Genetic improved fitness {}'.format(min_fitness[min_sum_index]))
    if filename:
        plot_cities(min_path[min_sum_index], min_fitness[min_sum_index], filename)
    return min_path[min_sum_index], min_fitness[min_sum_index]


def prim_euler_solve(distance_matrix, cities_list, filename=None):
    m_copy = np.empty_like(distance_matrix)
    m_copy[:] = distance_matrix
    t1 = time.time()
    print('Prim-Euler start at {}'.format(t1))
    path = prim_euler.solve(m_copy)
    t2 = time.time() - t1
    print('Prim-Euler ended after {}'.format(t2))
    cities_way = [cities_list[path[i]] for i in np.arange(0, len(cities_list), 1)]
    cities_way.append(cities_way[0])
    fitness = calc_fitness(cities_way)
    print('Prim-Euler fitness {}'.format(fitness))
    if filename:
        plot_cities(cities_way, fitness, filename)
    return cities_way, fitness


def aco(distance_matrix, cities_list, filename=None):
    m_copy = np.empty_like(distance_matrix)
    m_copy[:] = distance_matrix
    t1 = time.time()
    print('ACO start at {}'.format(t1))
    path = ant.solve(cities_list, distance_matrix)
    t2 = time.time() - t1
    print('ACO ended after {}'.format(t2))
    path.append(path[0])
    fitness = calc_fitness(path)
    print('ACO fitness {}'.format(fitness))
    if filename:
        plot_cities(path, fitness, filename)
    return path, fitness


def generate_data():
    cities_count_list = [10, 20, 30, 40, 50]

    for i in cities_count_list:
        cities, matrix = generate(i, i ** 2)
        write_to_file(cities, 'data/data-{}.txt'.format(i))


def test_with_plot():
    coords, distance_matrix = from_file('data/data-{}.txt'.format(50))
    prim_euler_solve(distance_matrix, coords, 'pe-50.png')
    aco(distance_matrix, coords, 'aco-50.png')
    greedy(distance_matrix, coords, 'greedy-50.png')
    greedy_improved(distance_matrix, coords, 'greedy-improved-50.png')
    anneal(coords, 'anneal-50.png')
    genetic(coords, 'genetic-50.png')


def test_compare():
    cities_count_list = [10, 20, 30, 40, 50]
    with open('results.txt', 'w') as f:
        for count in cities_count_list:
            coords, distance_matrix = from_file('data/data-{}.txt'.format(count))
            f.write('{}'.format(count))
            _, fitness = aco(distance_matrix, coords)
            f.write('ACO {}\n'.format(fitness))
            _, fitness = prim_euler_solve(distance_matrix, coords)
            f.write('Prim-Euler {}\n'.format(fitness))
            _, fitness = greedy(distance_matrix, coords)
            f.write('Greedy {}\n'.format(fitness))
            _, fitness = greedy_improved(distance_matrix, coords)
            f.write('Greedy_improved {}\n'.format(fitness))
            _, fitness = anneal(coords)
            f.write('Anneal {}\n'.format(fitness))
            _, fitness = genetic(coords)
            f.write('Genetic {}\n'.format(fitness))
            print('Done {}'.format(count))


def draw_compare():
    result = {
        'ACO': {
            'color': 'b',
            'title': 'Муравьиный',
            'results': {
                'x': [],
                'y': []
            }
        },
        'Prim-Euler': {
            'color': 'r',
            'title': 'Прима-Эйлера',
            'results': {
                'x': [],
                'y': []
            }
        },
        'Greedy': {
            'color': 'g',
            'title': 'Жадный',
            'results': {
                'x': [],
                'y': []
            }
        },
        'Greedy_improved': {
            'color': 'y',
            'title': 'Жадный (модификация)',
            'results': {
                'x': [],
                'y': []
            }
        },
        'Anneal': {
            'color': 'k',
            'title': 'Имитация отжига',
            'results': {
                'x': [],
                'y': []
            }
        },
        'Genetic': {
            'color': 'fuchsia',
            'title': 'Генетичесикий',
            'results': {
                'x': [],
                'y': []
            }
        },
    }
    algs_count = 6
    experiments_count = 5
    with open('test_results.txt', 'r') as f:
        for _ in np.arange(experiments_count):
            size = int(f.readline())
            for _ in np.arange(algs_count):
                [name, value] = f.readline().split(' ')
                in_dict = result[name]
                if in_dict is None:
                    print('Error!')
                    return
                in_dict['results']['x'].append(size)
                in_dict['results']['y'].append(float(value))
    handlers = []
    for alg in result:
        data = result[alg]
        handle, = plt.plot(data['results']['x'], data['results']['y'], color=data['color'], linewidth='2',
                           label=data['title'])
        handlers.append(handle)
    plt.legend(handles=handlers, loc=3)
    plt.grid(True)
    plt.show()


def draw_compare_big_data():
    result = {
        'Prim-Euler': {
            'color': 'r',
            'title': 'Прима-Эйлера',
            'results': {
                'x': [0],
                'y': [0.0]
            }
        },
        'Greedy': {
            'color': 'g',
            'title': 'Жадный',
            'results': {
                'x': [0],
                'y': [0.0]
            }
        },
        'Greedy_improved': {
            'color': 'y',
            'title': 'Жадный (модификация)',
            'results': {
                'x': [0],
                'y': [0.0]
            }
        },
        'Anneal': {
            'color': 'k',
            'title': 'Имитация отжига',
            'results': {
                'x': [0],
                'y': [0.0]
            }
        },
        'Genetic': {
            'color': 'fuchsia',
            'title': 'Генетичесикий',
            'results': {
                'x': [0],
                'y': [0.0]
            }
        },
        'ACO': {
            'color': 'b',
            'title': 'Муравьиный',
            'results': {
                'x': [0],
                'y': [0.0]
            }
        },
    }
    cities_count_list = [10, 20, 30, 40, 50]
    with open('check.txt', 'r') as f:
        for count in cities_count_list:
            values = f.readline().split(' ')
            result['Prim-Euler']['results']['x'].append(count)
            result['Prim-Euler']['results']['y'].append(float(values[0]))
            result['Greedy']['results']['x'].append(count)
            result['Greedy']['results']['y'].append(float(values[1]))
            result['Greedy_improved']['results']['x'].append(count)
            result['Greedy_improved']['results']['y'].append(float(values[2]))
            result['Anneal']['results']['x'].append(count)
            result['Anneal']['results']['y'].append(float(values[3]))
            result['Genetic']['results']['x'].append(count)
            result['Genetic']['results']['y'].append(float(values[4]))
            result['ACO']['results']['x'].append(count)
            result['ACO']['results']['y'].append(float(values[5]))
    handlers = []
    for alg in result:
        data = result[alg]
        handle, = plt.plot(data['results']['x'], data['results']['y'], color=data['color'], linewidth='2',
                           label=data['title'])
        handlers.append(handle)
    plt.legend(handles=handlers, loc=3)
    plt.grid(True)
    plt.show()


def generate_big_data():
    cities_count_list = [10, 20, 30, 40, 50]
    for count in cities_count_list:
        for i in np.arange(10):
            cities, matrix = generate(count, count * count)
            write_to_file(cities, 'test/out-{}-{}.txt'.format(count, i))


def compare_big_data():
    with open('check.txt', 'w') as f:
        cities_count_list = [10, 20, 30, 40, 50]
        for count in cities_count_list:
            t = np.zeros(5)
            for i in np.arange(10):
                coords, distance_matrix = from_file('test/out-{}-{}.txt'.format(count, i))
                _, fitness = prim_euler_solve(distance_matrix, coords)
                t[0] += fitness
                _, fitness = greedy(distance_matrix, coords)
                t[1] += fitness
                _, fitness = greedy_improved(distance_matrix, coords)
                t[2] += fitness
                _, fitness = anneal(coords)
                t[3] += fitness
                _, fitness = genetic(coords)
                t[4] += fitness
                print('Done {}'.format(i))
            f.write(' '.join(map(str, t / 10)))
            f.write('\n')
            print(t / 10)


def compare_greedy():
    result = {
        'Greedy': {
            'color': 'g',
            'title': 'Жадный',
            'results': {
                'x': [0],
                'y': [0.0]
            }
        },
        'Greedy_improved': {
            'color': 'y',
            'title': 'Жадный (модификация)',
            'results': {
                'x': [0],
                'y': [0.0]
            }
        }
    }
    cities_count_list = [10, 20, 30, 40, 50]
    for count in cities_count_list:
        t = np.zeros(2)
        for i in np.arange(10):
            coords, distance_matrix = from_file('test/out-{}-{}.txt'.format(count, i))
            _, fitness = greedy(distance_matrix, coords)
            t[0] += fitness
            _, fitness = greedy_improved(distance_matrix, coords)
            t[1] += fitness
            print('Done {}'.format(i))
        mid = t / 10
        result['Greedy']['results']['x'].append(count)
        result['Greedy']['results']['y'].append(mid[0])
        result['Greedy_improved']['results']['x'].append(count)
        result['Greedy_improved']['results']['y'].append(mid[1])
        print(t / 10)
    handlers = []
    for alg in result:
        data = result[alg]
        handle, = plt.plot(data['results']['x'], data['results']['y'], color=data['color'], linewidth='2',
                           label=data['title'])
        handlers.append(handle)
    plt.legend(handles=handlers, loc=3)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # draw_compare()
    # generate_big_data()
    # compare_big_data()
    # draw_compare_big_data()
    compare_greedy()
