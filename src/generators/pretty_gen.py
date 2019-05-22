import numpy as np
import math
from src.models.City import City


def generate(size, max_len):
    max_coord = math.ceil(max_len / np.sqrt(2))
    x = np.random.uniform(1, max_coord, size)
    y = np.random.uniform(1, max_coord, size)
    cities = []
    m = np.zeros([size, size])
    for i in np.arange(0, size, 1):
        for j in np.arange(0, size, 1):
            if i != j:
                m[i, j] = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            else:
                m[i, j] = float('inf')
    for i in np.arange(0, size, 1):
        city = City('City' + str(i), x[i], y[i])
        cities.append(city)
    for i in np.arange(0, size, 1):
        cities[i].calculate_distances(cities)
    return cities, m


def write_to_file(cities, filename='out.txt'):
    size = len(cities)
    with open(filename, 'w') as out:
        for i in np.arange(0, size, 1):
            city = cities[i]
            out.write('{}\n'.format(city))


def from_file(filename):
    cities = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            [name, x, y] = line.split(' ')
            city = City(name, float(x), float(y))
            cities.append(city)
    size = len(cities)
    m = np.zeros([size, size])
    for i in np.arange(0, size, 1):
        cities[i].calculate_distances(cities)
        for j in np.arange(0, size, 1):
            if i != j:
                m[i, j] = np.sqrt((cities[i].x - cities[j].x) ** 2 + (cities[i].y - cities[j].y) ** 2)
            else:
                m[i, j] = float('inf')
    return cities, m
