import numpy as np
from random import random, choice

MAX_TIME = float('inf')
MAX_ANTS = float('inf')
ALPHA = 1  # вес фермента
BETA = 5  # коэффициент эвристики, влияние априорных знаний(1/d, где d - растояние)
RHO = .5  # Интенсивность. Коф. испарение равен 1 - RHO. По результатам тестов лучше использовать >= 0.5
QVAL = 100  # Кол. феромонов на один проход
PR = 0.01  # вероятность полностью случайного перехода


class Ant(object):

    def __init__(self, start_city):
        self.cur_city = start_city
        self.path = [start_city]
        self.tour_length = 0.

    def move_to_city(self, city):
        self.path.append(city)
        self.tour_length += DISTANCE[self.cur_city][city]
        if len(self.path) == MAX_CITIES:
            self.tour_length += DISTANCE[self.path[-1]][self.path[0]]
        self.cur_city = city

    def can_move(self):
        return len(self.path) < MAX_CITIES

    def reset(self, city):
        self.cur_city = city
        self.path = [city]
        self.tour_length = 0.


def get_random(l):
    r = random()
    cur_probability = 0
    cur_val = None

    for val, probability in l:
        cur_val = val
        cur_probability += probability
        if r <= cur_probability:
            break

    return cur_val


ANTS = []  # [MAX_ANTS]
CITIES = []  # [MAX_CITIES]
DISTANCE = []  # [MAX_CITIES][MAX_CITIES]
PHEROMONE = []  # [MAX_CITIES][MAX_CITIES]
BEST = float('inf')
BEST_ANT = None


def init(cities, distance_matrix):
    global DISTANCE, PHEROMONE, CITIES, ANTS, MAX_CITIES, MAX_TIME, MAX_ANTS
    MAX_CITIES = len(cities)
    MAX_TIME = 500 * MAX_CITIES
    MAX_ANTS = MAX_CITIES * MAX_CITIES
    init_pheromone = 1.0 / MAX_CITIES
    PHEROMONE = np.full((MAX_CITIES, MAX_CITIES), init_pheromone)

    CITIES = cities
    DISTANCE = distance_matrix

    to = 0
    for _ in np.arange(0, MAX_ANTS, 1):
        ANTS.append(Ant(to))
        to += 1
        to = to % MAX_CITIES


def ant_product(from_city, to_city, ph=None):
    global DISTANCE, PHEROMONE, ALPHA, BETA
    ph = ph or PHEROMONE[from_city][to_city]
    return (ph ** ALPHA) * \
           ((1. / DISTANCE[from_city][to_city]) ** BETA)


def select_next_city(ant):
    global MAX_CITIES, PHEROMONE, DISTANCE
    denom = 0.
    not_visited = []

    for to in np.arange(0, MAX_CITIES, 1):
        if to not in ant.path:
            ap = ant_product(ant.cur_city, to)
            not_visited.append((to, ap))
            denom += ap
    r = random()
    not_visited = [(val, ap / denom) for (val, ap) in not_visited]
    if r < PR:
        to = choice(not_visited)
        return to[0]
    to = get_random(not_visited)
    return to


def simulate_ants():
    moving = 0

    for ant in ANTS:
        if ant.can_move():
            ant.move_to_city(select_next_city(ant))
            moving += 1

    return moving


def update_trails():
    global MAX_CITIES, PHEROMONE, RHO, ANTS

    for ant in ANTS:
        pheromove_amount = QVAL / ant.tour_length

        for i in np.arange(0, MAX_CITIES, 1):
            if i == MAX_CITIES - 1:
                from_city = ant.path[i]
                to_city = ant.path[0]
            else:
                from_city = ant.path[i]
                to_city = ant.path[i + 1]
            assert from_city != to_city
            PHEROMONE[from_city][to_city] = PHEROMONE[from_city][to_city] * (1 - RHO) + pheromove_amount
            PHEROMONE[to_city][from_city] = PHEROMONE[from_city][to_city]


def restart_ants():
    global ANTS, BEST, BEST_ANT, MAX_CITIES
    to = 0

    for ant in ANTS:
        if ant.tour_length < BEST:
            BEST = ant.tour_length
            BEST_ANT = ant

        ant.reset(to)
        to += 1
        to = to % MAX_CITIES


def solve(cities, distance_matrix):
    init(cities, distance_matrix)
    cur_time = 0
    path = []
    while cur_time < MAX_TIME:
        cur_time += 1
        if cur_time % 100 == 0:
            print('time:', cur_time, 'of', MAX_TIME)

        if simulate_ants() == 0:
            update_trails()
            cur_time != MAX_TIME and restart_ants()
    for i in BEST_ANT.path:
        path.append(CITIES[i])
    return path
