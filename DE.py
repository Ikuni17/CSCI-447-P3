'''
CSCI 447: Project 3
Group 28: Trent Baker, Logan Bonney, Bradley White
November 13, 2017

This file contains the functionality for the Differential Evolution Algorithm. Some of which is inherited from GA.
'''

import GA
import MLP
import matplotlib.pyplot as plt
import pandas
import random
import statistics as stats
import time


# Mutate each individual
def mutate(population, beta):
    trial_vectors = []
    diff1 = 0
    diff2 = 0

    # Iterate through the whole population
    for i in range(len(population)):
        while (i == diff1 or i == diff2 or diff1 == diff2):
            # Choose two individuals at random and get the difference between them
            diff1 = random.randrange(0, len(population) - 1)
            diff2 = random.randrange(0, len(population) - 1)
            diff_list = diff(population[diff1], population[diff2])

        # Create a trial vector based on the individual and the difference of the two random individuals altered by beta
        trial_vectors.append(population[i] + [x * beta for x in diff_list])
    return trial_vectors


# Create a child vector from a parent and trial vector
def crossover(trial_vector, parent, crossover_rate):
    # Force crossover for at least one point
    point = random.randrange(0, len(parent) - 1)
    child = list(parent)
    child[point] = trial_vector[point]

    # Try to crossover more points
    for j in range(len(parent) - 1):
        if random.uniform(0, 1) < crossover_rate and j != point:
            child[j] = trial_vector[point]
    return child


# Choose between keeping a parent or replacing with their child
def selection(nn, child, parent):
    # Evaluate both
    child_error = GA.evaluate(nn, child)
    parent_error = GA.evaluate(nn, parent)

    # Choose the best between parents and child, also return their error for output at the end
    if child_error < parent_error:
        return (child, child_error)
    else:
        return (parent, parent_error)


# Return the difference of two individuals
def diff(first, second):
    return [item for item in first if item not in second]


# Train a neural network with DE
def train(nn, max_gen, pop_size, crossover_rate, beta, process_ID=0):
    generation = 0
    mean_error = []
    # Initialize the population with the GA function
    population = GA.init_population(nn, pop_size)

    # print("Starting DE training at {0}".format(time.ctime(time.time())))

    # Loop until maximum generations
    while generation < max_gen:
        # Generate trial vectors
        trial_vectors = mutate(population, beta)

        # Track this generations error
        gen_error = []
        # Generate children from trial vectors
        for i in range(len(trial_vectors)):
            child = crossover(trial_vectors[i], population[i], crossover_rate)
            # Choose between parent and child
            temp_tuple = selection(nn, child, population[i])
            population[i] = temp_tuple[0]
            gen_error.append(temp_tuple[1])

        # Get the mean error of the population and track it for each generation
        temp_mean = stats.mean(gen_error)
        mean_error.append(temp_mean)

        # Print the progress periodically
        if (generation % 100 == 0):
            print("DE{2}: Generation {0}, Mean Error: {1}".format(generation, temp_mean, process_ID))

        # Move to the next generation
        generation += 1

    # print("Finished DE training at {0}".format(time.ctime(time.time())))
    return mean_error


# Test run if this file is ran on its own
if __name__ == '__main__':
    af_path = 'datasets\\converted\\airfoil.csv'
    df = pandas.read_csv(af_path, header=None)
    training_data = df.values.tolist()
    nn = MLP.MLP(len(training_data[0]) - 1, 1, 10, training_data)
    mean_error = train(nn, 2000, 100, 0.5, 0.1)

    plt.plot(mean_error, label='DE')
    plt.xlabel('Generation')
    plt.ylabel('Mean Squared Error')
    plt.yscale('log')
    plt.title('DE')
    plt.legend()
    plt.show()
