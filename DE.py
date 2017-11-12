'''
CSCI 447: Project 3
Group 28: Trent Baker, Logan Bonney, Bradley White
November 13, 2017
'''

import GA
import random
import numpy as np
import MLP
import rosen_generator as rosen

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
    child = list(parent)

    # Force crossover for at least one point
    child[point] = trial_vector[point]
    for j in range(len(parent)-1):
        if random.uniform(0, 1) < crossover_prob and j != point:
            child[j] = trial_vector[point]
    return child


def selection(child, parent):
    child_error = GA.evaluate(nn, child)
    parent_error = GA.evaluate(nn, parent)
    # Choose the best between parents adn children
    if child_error < parent_error:
        return child
    else:
        return parent


def diff(first, second):
    return [item for item in first if item not in second]


def train():
    generation = 0
    max_gen = 2000
    pop_size = 100
    population = GA.init_population(nn, pop_size)
    while generation < max_gen:
        # Generate trial vectors
        trial_vectors = mutate(population)
        #print(trial_vectors)
        # generate children from trial_vectors
        for i in range(len(trial_vectors)):
            child = crossover(trial_vectors[i], population[i])
            population[i] = selection(child, population[i])
        if(generation % 5 == 0):
            print("Generation {0}, Error: {1}".format(generation, GA.evaluate(nn, population[0])))
        generation += 1

if __name__ == '__main__':
    num_inputs = 2
    data = MLP.read_csv('concrete')
    training_data = rosen.generate(0, num_inputs)
    nn = MLP.MLP(num_inputs, 1, 10, training_data)
    train()
