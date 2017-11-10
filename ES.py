'''
CSCI 447: Project 3
Group 28: Trent Baker, Logan Bonney, Bradley White
November 13, 2017
'''

import GA
import MLP
import random
import math
import rosen_generator as rosen
import time
import statistics as stats


def add_sigmas(pop_size, individual_size):
    sigmas = []

    for i in range(pop_size):
        sigmas.append([])
        for j in range(individual_size):
            sigmas[i].append(random.gauss(1, 1))

    return sigmas


# p.218 second bullet
def update_sigmas(sigmas):
    tau = (1 / math.sqrt(2 * math.sqrt(len(sigmas[0]))))
    tau_prime = (1 / math.sqrt(2 * len(sigmas[0])))

    for i in range(len(sigmas)):
        for j in range(len(sigmas[i])):
            sigmas[i][j] = sigmas[i][j] * math.exp((tau_prime * random.gauss(0, 1)) + (tau * random.gauss(0, 1)))

    return sigmas


def train(nn, max_gen, pop_size, num_children, crossover_rate, mutation_rate):
    generation = 0
    population = GA.init_population(nn, pop_size)
    sigmas = add_sigmas(pop_size, len(population[0]))
    heat_size = 10

    print("Starting ES training at {0}".format(time.ctime(time.time())))

    # TODO stop when converged?
    while (generation < max_gen):





        # Select the best parents and use them to produce pop_size children and overwrite the entire population
        population = GA.crossover_multipoint(GA.rank_selection(nn, population, pop_size), pop_size, crossover_rate)
        # population = crossover_multipoint(tournament_selection(nn, population, heat_size), pop_size)

        # Try to mutate each child
        for i in range(len(population)):
            population[i] = GA.mutate(population[i], mutation_rate)

        if (generation % 5 == 0):
            print("Generation {0}, Mean Error: {1}".format(generation, stats.mean(GA.pop_error)))
        # Move to the next generation
        generation += 1

    print("Finished ES training at {0}".format(time.ctime(time.time())))


if __name__ == '__main__':
    num_inputs = 2
    training_data = rosen.generate(0, num_inputs)
    nn = MLP.MLP(num_inputs, 1, 100, training_data)
    train(nn, 2000, 200, 20, 0.5, 0.1)
