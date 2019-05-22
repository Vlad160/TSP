import random
import copy
import time
# Original: https://github.com/maoaiz/tsp-genetic-python
list_of_cities = []

# probability that an individual Route will mutate
k_mut_prob = 0.30

k_n_generations = 550
# Population size of 1 generation (RoutePop)
k_population_size = 170

# Size of the tournament selection. 
tournament_size = 8

# If elitism is True, the best from one generation will carried over to the next.
elitism = True


# Route Class
class Route(object):

    def __init__(self):
        self.route = sorted(list_of_cities, key=lambda *args: random.random())
        self.recalc_rt_len()

    def recalc_rt_len(self):
        self.length = 0.0
        for city in self.route:
            next_city = self.route[self.route.index(city) - len(self.route) + 1]
            dist_to_next = city.distance_to[next_city.name]
            self.length += dist_to_next

    def pr_cits_in_rt(self, print_route=False):
        cities_str = ''
        for city in self.route:
            cities_str += city.name + ','
        cities_str = cities_str[:-1]  # chops off last comma
        if print_route:
            print('    ' + cities_str)

    def pr_vrb_cits_in_rt(self):
        cities_str = '|'
        for city in self.route:
            cities_str += str(city.x) + ',' + str(city.y) + '|'
        print(cities_str)

    def is_valid_route(self):
        for city in list_of_cities:
            # helper function defined up to
            if self.count_mult(self.route, lambda c: c.name == city.name) > 1:
                return False
        return True

    def count_mult(self, seq, pred):
        return sum(1 for v in seq if pred(v))


# Contains a population of Route() objects
class RoutePop(object):

    def __init__(self, size, initialise):
        self.rt_pop = []
        self.fittest = []
        self.size = size
        # If we want to initialise a population.rt_pop:
        if initialise:
            for x in range(0, size):
                new_rt = Route()
                self.rt_pop.append(new_rt)
            self.get_fittest()

    def get_fittest(self):
        sorted_list = sorted(self.rt_pop, key=lambda x: x.length, reverse=False)
        self.fittest = sorted_list[0]
        return self.fittest


class GA(object):

    def crossover(self, parent1, parent2):

        child_rt = Route()

        for x in range(0, len(child_rt.route)):
            child_rt.route[x] = None

        start_pos = random.randint(0, len(parent1.route))
        end_pos = random.randint(0, len(parent1.route))

        if start_pos < end_pos:
            for x in range(start_pos, end_pos):
                child_rt.route[x] = parent1.route[x]
        elif start_pos > end_pos:
            # do it in the end-->start order
            for i in range(end_pos, start_pos):
                child_rt.route[i] = parent1.route[i]

        for i in range(len(parent2.route)):
            # if parent2 has a city that the child doesn't have yet:
            if not parent2.route[i] in child_rt.route:
                # it puts it in the first 'None' spot and breaks out of the loop.
                for x in range(len(child_rt.route)):
                    if child_rt.route[x] is None:
                        child_rt.route[x] = parent2.route[i]
                        break

        child_rt.recalc_rt_len()
        return child_rt

    def mutate(self, route_to_mut):

        if random.random() < k_mut_prob:

            mut_pos1 = random.randint(0, len(route_to_mut.route) - 1)
            mut_pos2 = random.randint(0, len(route_to_mut.route) - 1)

            if mut_pos1 == mut_pos2:
                return route_to_mut

            city1 = route_to_mut.route[mut_pos1]
            city2 = route_to_mut.route[mut_pos2]

            route_to_mut.route[mut_pos2] = city1
            route_to_mut.route[mut_pos1] = city2

        route_to_mut.recalc_rt_len()

        return route_to_mut

    def tournament_select(self, population):

        tournament_pop = RoutePop(size=tournament_size, initialise=False)

        for i in range(tournament_size - 1):
            tournament_pop.rt_pop.append(random.choice(population.rt_pop))

        return tournament_pop.get_fittest()

    def evolve_population(self, init_pop):

        descendant_pop = RoutePop(size=init_pop.size, initialise=True)

        elitismOffset = 0

        if elitism:
            descendant_pop.rt_pop[0] = init_pop.fittest
            elitismOffset = 1

        for x in range(elitismOffset, descendant_pop.size):
            tournament_parent1 = self.tournament_select(init_pop)
            tournament_parent2 = self.tournament_select(init_pop)

            tournament_child = self.crossover(tournament_parent1, tournament_parent2)

            descendant_pop.rt_pop[x] = tournament_child

        for route in descendant_pop.rt_pop:
            if random.random() < k_mut_prob:
                self.mutate(route)

        descendant_pop.get_fittest()

        return descendant_pop


class App(object):

    def __init__(self, n_generations, pop_size):

        self.n_generations = n_generations
        self.pop_size = pop_size
        self.best_route = []
        print("Calculating GA_loop")
        self.GA_loop(n_generations, pop_size)

    def GA_loop(self, n_generations, pop_size):

        start_time = time.time()

        print("Creates the population:")
        the_population = RoutePop(pop_size, True)
        print("Finished Creation of the population")

        if the_population.fittest.is_valid_route() == False:
            raise NameError('Multiple cities with same name. Check cities.')
            return

        initial_length = the_population.fittest.length

        best_route = Route()

        for x in range(1, n_generations):

            the_population = GA().evolve_population(the_population)

            if the_population.fittest.length < best_route.length:
                best_route = copy.deepcopy(the_population.fittest)

        end_time = time.time()

        print('Finished evolving {0} generations.'.format(n_generations))
        print("Elapsed time was {0:.1f} seconds.".format(end_time - start_time))
        print(' ')
        print('Initial best distance: {0:.2f}'.format(initial_length))
        print('Final best distance:   {0:.2f}'.format(best_route.length))
        print('The best route went via:')
        best_route.pr_cits_in_rt(print_route=True)
        self.best_route = best_route


def random_cities(cities):
    global list_of_cities
    list_of_cities = []
    for city in cities:
        list_of_cities.append(city)
    # create and run an application instance:
    app = App(n_generations=k_n_generations, pop_size=k_population_size)
    return app.best_route.route
