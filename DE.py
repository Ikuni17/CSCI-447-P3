import GA
import random
import numpy as np

BETA = 0.1
crossover_prob = 0.1

def mutate(population):
    trial_vectors = []
    diff1 = 0
    diff2 = 0
    for i in range(len(population)):
        while (i == diff1 or i == diff2 or diff1 == diff2):
            diff1 = random.randrange(0, len(population) - 1)
            diff2 = random.randrange(0, len(population) - 1)
            diff_list = diff(population[diff1], population[diff2])
        trial_vectors.append(population[i] + [x * BETA for x in diff_list])
    return trial_vectors


def crossover(trial_vector, parent):
    point = random.randrange(0, len(parent)-1)
    child = parent

    # Force crossover for at least one point
    child[point] = trial_vector[point]
    for j in range(len(parent)-1):
        if random.uniform(0, 1) < crossover_prob and j != point:
            child[j] = trial_vector[point]
    return child


def selection(child, parent):
    child_perf = GA.evaluate(child)
    parent_perf = GA.evaluate(parent)
    # Choose the best between parents adn children
    if child_perf > parent_perf:
        return child
    else:
        return parent


def diff(first, second):
    return [item for item in first if item not in second]


def train():
    generation = 0
    max_gen = 2000
    pop_size = 100
    population = GA.init_population(pop_size)
    while generation < max_gen:
        # Generate trial vectors
        trial_vectors = mutate(population)
        # generate children from trial_vectors
        for i in range(len(trial_vectors)):
            child = crossover(trial_vectors[i], population[i])
            population[i] = selection(child, population[i])


if __name__ == '__main__':
    train()
