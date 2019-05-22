import numpy as np


class City(object):

    def __init__(self, name, x, y, distance_to=None):
        self.name = name
        self.x = self.graph_x = x
        self.y = self.graph_y = y
        self.distance_to = {self.name: 0.0}
        if distance_to:
            self.distance_to = distance_to

    def calculate_distances(self, list_of_cities):
        for city in list_of_cities:
            tmp_dist = self.point_dist(self.x, self.y, city.x, city.y)
            self.distance_to[city.name] = tmp_dist

    def point_dist(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def __str__(self):
        return '{} {} {}'.format(self.name, self.x, self.y)
