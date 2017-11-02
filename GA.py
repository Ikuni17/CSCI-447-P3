'''import itertools

thing = [[['A'], ['B'], ['C']],[['D'], ['E'], ['F']]]

thing = list(itertools.chain.from_iterable(thing))
thing = list(itertools.chain.from_iterable(thing))
print(thing)'''

import MLP

crossover_rate = 0.5
mutation_rate = 0.1
evaluation = []

def init_population(size):
    population = []
    weights = MLP.get_nn()
    for i in range(size):
        population.append(weights)

    return population

def evaluate():
    pass

def crossover(p1, p2):
    global crossover_rate
    pass

def mutate(child):
    global mutation_rate
    pass

def selection(population):
    pass

def train():
    generation = 0