import numpy as np

from src.models.City import City


def generate(size, max_coord):
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
