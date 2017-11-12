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
import statistics as stats
import matplotlib.pyplot as plt
import time


def mutate(population, beta):
    trial_vectors = []
    diff1 = 0
    diff2 = 0
    for i in range(len(population)):
        while (i == diff1 or i == diff2 or diff1 == diff2):
            diff1 = random.randrange(0, len(population) - 1)
            diff2 = random.randrange(0, len(population) - 1)
            diff_list = diff(population[diff1], population[diff2])
        trial_vectors.append(population[i] + [x * beta for x in diff_list])
    return trial_vectors


def crossover(trial_vector, parent, crossover_rate):
    point = random.randrange(0, len(parent) - 1)
    child = list(parent)

    # Force crossover for at least one point
    child[point] = trial_vector[point]
    for j in range(len(parent) - 1):
        if random.uniform(0, 1) < crossover_rate and j != point:
            child[j] = trial_vector[point]
    return child


def selection(nn, child, parent):
    child_error = GA.evaluate(nn, child)
    parent_error = GA.evaluate(nn, parent)
    # Choose the best between parents adn children
    if child_error < parent_error:
        return (child, child_error)
    else:
        return (parent, parent_error)


def diff(first, second):
    return [item for item in first if item not in second]


def train(nn, max_gen, pop_size, crossover_rate, beta, process_ID=0):
    generation = 0
    mean_error = []
    population = GA.init_population(nn, pop_size)

    print("Starting DE training at {0}".format(time.ctime(time.time())))
    while generation < max_gen:
        # Generate trial vectors
        trial_vectors = mutate(population, beta)
        # print(trial_vectors)
        # generate children from trial_vectors
        temp_vector = []
        for i in range(len(trial_vectors)):
            child = crossover(trial_vectors[i], population[i], crossover_rate)
            temp_tuple = selection(nn, child, population[i])
            population[i] = temp_tuple[0]
            temp_vector.append(temp_tuple[1])

        temp_mean = stats.mean(temp_vector)
        mean_error.append(temp_mean)

        if (generation % 1000 == 0):
            print("DE{2}: Generation {0}, Mean Error: {1}".format(generation, temp_mean, process_ID))
        generation += 1

    print("Finished DE training at {0}".format(time.ctime(time.time())))
    return mean_error


if __name__ == '__main__':
    num_inputs = 2
    data = MLP.read_csv('concrete')
    training_data = rosen.generate(0, num_inputs)
    nn = MLP.MLP(num_inputs, 1, 10, training_data)
    mean_error = train(nn, 2000, 100, 0.5, 0.1)

    plt.plot(mean_error, label='DE')
    plt.xlabel('Generation')
    plt.ylabel('Mean Squared Error')
    plt.yscale('log')
    plt.title('DE')
    plt.legend()
    plt.show()
