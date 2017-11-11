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


def add_sigmas(individual):
    for i in range(len(individual)):
        individual.append(random.gauss(0, 1))
    return individual

def init_population(nn, pop_size):
    population = GA.init_population(nn, pop_size)
    for i in range(len(population)):
        population[i] = add_sigmas(population[i])
    return population


# p.218 second bullet
def update_sigmas(individual):
    sigmas = individual[len(individual)/2:]
    tau = (1 / math.sqrt(2 * math.sqrt(len(sigmas))))
    tau_prime = (1 / math.sqrt(2 * len(sigmas)))

    for i in range(len(sigmas)):
        sigmas[i] *= math.exp((tau_prime * random.gauss(0, 1)) + (tau * random.gauss(0, 1)))

    individual = individual[:len(individual)/2]
    for sigma in sigmas:
        individual.append(sigma)

    return individual


def apply_sigmas(child):
    sigmas = child[len(individual)/2:]
        for index in range(len(sigmas)):
            delta = sigma[index] * random.gauss(0, 1)
            child[index] += delta
    return child


def mutate(individual):
    update_sigmas(individual)
    apply_sigmas(individual)



def rank_selection(nn, population, pop_size):
    pop_error = []

    rank_weights = []
    for individual in population:
        fitness = evaluate(nn, individual[:len(individual)/2])
        rank_weights.append(1 / fitness)
        pop_error.append(fitness)

    return (random.choices(population, rank_weights, k=pop_size), pop_error)

def train(nn, max_gen, pop_size, num_children, crossover_rate, mutation_rate):
    generation = 0
    population = GA.init_population(nn, pop_size)
    sigmas = gen_sigmas(pop_size, len(population[0]))
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


def print_pop(population):
    for element in population:
        print(str(element))

if __name__ == '__main__':
    num_inputs = 2
    training_data = rosen.generate(0, num_inputs)
    nn = MLP.MLP(num_inputs, 1, 2, training_data)
    population = init_population(nn, 2)
    print_pop(population)
    # train(nn, 2000, 200, 20, 0.5, 0.1)
