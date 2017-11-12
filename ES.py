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
import matplotlib.pyplot as plt


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
    sigmas = individual[math.ceil(len(individual) / 2):]
    tau = (1 / math.sqrt(2 * math.sqrt(len(sigmas))))
    tau_prime = (1 / math.sqrt(2 * len(sigmas)))

    for i in range(len(sigmas)):
        sigmas[i] *= math.exp((tau_prime * random.gauss(0, 1)) + (tau * random.gauss(0, 1)))

    individual = individual[:math.floor(len(individual) / 2)]
    for sigma in sigmas:
        individual.append(sigma)

    return individual


def apply_sigmas(child):
    sigmas = child[math.ceil(len(child) / 2):]
    for index in range(len(sigmas)):
        delta = sigmas[index] * random.gauss(0, 1)
        child[index] += delta
    return child


def mutate(individual):
    return apply_sigmas(update_sigmas(individual))


def rank_selection(nn, population, pop_size):
    pop_error = []

    rank_weights = []
    for individual in population:
        fitness = GA.evaluate(nn, individual[:math.floor(len(individual) / 2)])
        rank_weights.append(1 / fitness)
        pop_error.append(fitness)

    return (random.choices(population, rank_weights, k=pop_size), pop_error)


def train(nn, max_gen, pop_size, num_children, crossover_rate, process_id=0):
    generation = 0
    population = init_population(nn, pop_size)
    mean_error = []
    heat_size = 10

    #print("Starting ES training at {0}".format(time.ctime(time.time())))

    # TODO stop when converged?
    while (generation < max_gen):
        children = GA.crossover_multipoint(population, num_children, crossover_rate)

        for i in range(num_children):
            children[i] = mutate(children[i])

        temp_tuple = rank_selection(nn, population + children, pop_size)
        population = temp_tuple[0]

        temp_mean = stats.mean(temp_tuple[1])
        mean_error.append(temp_mean)

        if (generation % 1000 == 0):
            print("ES{2}: Generation {0}, Mean Error: {1}".format(generation, temp_mean, process_id))

        # Move to the next generation
        generation += 1

    #print("Finished ES training at {0}".format(time.ctime(time.time())))
    return mean_error


def print_pop(population):
    for element in population:
        print(str(element))


if __name__ == '__main__':
    num_inputs = 2
    training_data = rosen.generate(0, num_inputs)
    nn = MLP.MLP(num_inputs, 1, 10, training_data)
    mean_error = train(nn, 2000, 100, 100, 0.5)

    plt.plot(mean_error, label='ES')
    plt.xlabel('Generation')
    plt.ylabel('Mean Squared Error')
    plt.yscale('log')
    plt.title('ES')
    plt.legend()
    plt.show()
