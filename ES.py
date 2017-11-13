'''
CSCI 447: Project 3
Group 28: Trent Baker, Logan Bonney, Bradley White
November 13, 2017

This file contains the functionality for the Evolutionary Strategy Algorithm. Specifically mu + lambda.
Some functionality is inherited from GA.
'''

import GA
import MLP
import math
import matplotlib.pyplot as plt
import pandas
import random
import statistics as stats
import time


# Initialize random sigma vector for an individual
def add_sigmas(individual):
    # Generate a sigma for each weight
    for i in range(len(individual)):
        individual.append(random.gauss(0, 1))
    return individual


# Initialize the population vectors by calling the function in GA, then appending the sigmas for each individual
def init_population(nn, pop_size):
    population = GA.init_population(nn, pop_size)
    for i in range(len(population)):
        population[i] = add_sigmas(population[i])
    return population


# Update the sigmas if an individual is mutated
def update_sigmas(individual):
    # Slice the sigmas off
    sigmas = individual[math.ceil(len(individual) / 2):]
    tau = (1 / math.sqrt(2 * math.sqrt(len(sigmas))))
    tau_prime = (1 / math.sqrt(2 * len(sigmas)))

    # Update each based on the defined equation
    for i in range(len(sigmas)):
        sigmas[i] *= math.exp((tau_prime * random.gauss(0, 1)) + (tau * random.gauss(0, 1)))

    # Slice the old sigmas off and append each new sigma on
    individual = individual[:math.floor(len(individual) / 2)]
    for sigma in sigmas:
        individual.append(sigma)

    # Return the individual with new sigmas
    return individual


# Apply the sigma to each weight of an individual and update the weight
def apply_sigmas(child):
    # Get the sigma slice
    sigmas = child[math.ceil(len(child) / 2):]

    # Get the delta for each weight and update it
    for index in range(len(sigmas)):
        delta = sigmas[index] * random.gauss(0, 1)
        child[index] += delta

    # Return the mutated child
    return child


# Update sigmas, then apply the sigmas to an individual
def mutate(individual):
    return apply_sigmas(update_sigmas(individual))


# Same rank selection as GA, but now we need to deal with slicing off the sigmas
# Rank each individual based on their fitness, then choose a pop_size population from them probabilistically
def rank_selection(nn, population, pop_size):
    # Keep track of the whole population's fitness for output later
    pop_error = []
    rank_weights = []

    # Evaluate each individual
    for individual in population:
        # Get the fitness for this individual without sigmas
        fitness = GA.evaluate(nn, individual[:math.floor(len(individual) / 2)])
        # Get their ranking
        rank_weights.append(1 / fitness)
        # Added their error to the vector
        pop_error.append(fitness)

    # Return a pop_size population, choosen probabilistically based on their weights, and the population error vector
    return (random.choices(population, rank_weights, k=pop_size), pop_error)


# Train a neural network with ES
def train(nn, max_gen, pop_size, num_children, crossover_rate, process_id=0):
    generation = 0
    population = init_population(nn, pop_size)
    mean_error = []

    # print("Starting ES training at {0}".format(time.ctime(time.time())))

    # Loop until maximum generations
    while (generation < max_gen):
        # Create lamba children (num_children)
        children = GA.crossover_multipoint(population, num_children, crossover_rate)

        # Mutate each child
        for i in range(num_children):
            children[i] = mutate(children[i])

        # Choose the best from the population and all the children and create a pop_size population
        temp_tuple = rank_selection(nn, population + children, pop_size)
        population = temp_tuple[0]

        # Get the mean error of the population and track it for each generation
        temp_mean = stats.mean(temp_tuple[1])
        mean_error.append(temp_mean)

        # Print the progress periodically
        if (generation % 100 == 0):
            print("ES{2}: Generation {0}, Mean Error: {1}".format(generation, temp_mean, process_id))

        # Move to the next generation
        generation += 1

    # print("Finished ES training at {0}".format(time.ctime(time.time())))
    return mean_error


# Test run if this file is ran on its own
if __name__ == '__main__':
    af_path = 'datasets\\converted\\airfoil.csv'
    df = pandas.read_csv(af_path, header=None)
    training_data = df.values.tolist()
    nn = MLP.MLP(len(training_data[0]) - 1, 1, 10, training_data)
    mean_error = train(nn, 2000, 100, 100, 0.5)

    plt.plot(mean_error, label='ES')
    plt.xlabel('Generation')
    plt.ylabel('Mean Squared Error')
    plt.yscale('log')
    plt.title('ES')
    plt.legend()
    plt.show()
