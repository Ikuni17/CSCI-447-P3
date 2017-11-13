'''
CSCI 447: Project 3
Group 28: Trent Baker, Logan Bonney, Bradley White
November 13, 2017

This file contains the functionality for the Genetic Algorithm. Some of which is inherited by ES and DE.
'''

import MLP
import matplotlib.pyplot as plt
import pandas
import random
import statistics as stats
import time


# Create a population of individual chromosomes, based on the size of the Neural Network
# The population is a matrix, where each individual is a vector of weights within it
def init_population(nn, size):
    population = []
    # Get the size of the network
    num_weights = len(nn.get_weights())
    # Randomize the individuals slightly, for genetic diversity
    for i in range(size):
        population.append(nn.generate_random_individual(num_weights))
    return population


# Determine the fitness of an individual by putting their weights into the neural network, forward propagating, and
# then calculating the individual
def evaluate(nn, individual):
    # Populate the network with this individual's weights
    nn.set_weights(individual)
    # Forward propagate
    nn.feedforward()

    # Return the error of this individual
    return nn.calc_avg_error()


# Takes a list of parents and produces (num_children) children with a random number of randomly selected slice points
def crossover_multipoint(parents, num_children, crossover_rate):
    children = []
    parent_num = 0

    # generate num_children children
    for index in range(num_children):
        children.append([])
        # decide each attribute for the current child
        for attribute in range(len(parents[0])):
            if random.random() < crossover_rate:
                # randomly select a parent from parents if we're within the crossover rate
                parent_num = int(random.random() * len(parents))
            # Add the attribute to the child
            children[index].append(parents[parent_num][attribute])
    return children


# Has a (mutation_rate) chance to change each attribute by a random number between -1.0 and 1.0
def mutate(child, mutation_rate):
    # Try to mutate each attribute
    for attribute in range(len(child)):
        if random.random() < mutation_rate:
            # If we are within the rate, creep the attribute
            child[attribute] += random.uniform(-1.0, 1.0)

    return child


# Rank each individual based on their fitness, then choose a pop_size population from them probabilistically
def rank_selection(nn, population, pop_size):
    # Keep track of the whole population's fitness for output later
    pop_error = []
    rank_weights = []

    # Evaluate each individual
    for individual in population:
        # Get the fitness for this individual
        fitness = evaluate(nn, individual)
        # Get their ranking
        rank_weights.append(1 / fitness)
        # Added their error to the vector
        pop_error.append(fitness)

    # Return a pop_size population, choosen probabilistically based on their weights, and the population error vector
    return (random.choices(population, rank_weights, k=pop_size), pop_error)


# Alternative selection algorithm, slower than rank_selection and unused currently
# Selects (num_select) individuals from (population) and holds a tournament with (heat_size) heats
def tournament_selection(nn, population, heat_size):
    num_select = len(population)
    selected = []

    # Select num_select individuals
    for i in range(num_select):
        # Randomly select heat_size individuals from the population
        heat = []
        for individual in range(heat_size):
            # Add a random individual to heat
            heat.append(population[int(random.random() * len(population))])

        # Find the best individual from heat and add it to selected
        # ASSUMING MINIMIZATION
        if heat != []:
            min = evaluate(nn, heat[0])
            min_index = 0

            # Evaluate each individual and keep the one with the lowest error
            for contestant in heat:
                temp_fitness = evaluate(nn, contestant)
                if temp_fitness < min:
                    min = temp_fitness
                    min_index = heat.index(contestant)

            selected.append(heat[min_index])
    return selected


# Train a neural network with GA
def train(nn, max_gen, pop_size, crossover_rate, mutation_rate, process_id=0):
    generation = 0
    population = init_population(nn, pop_size)
    mean_error = []

    # print("Starting GA training at {0}".format(time.ctime(time.time())))

    # Loop until maximum generations
    while (generation < max_gen):
        # Use rank selection to get a new pop_size population of "best" parents
        temp_tuple = rank_selection(nn, population, pop_size)
        # Perform crossover on the parents to create pop_size children and wipe the whole population
        population = crossover_multipoint(temp_tuple[0], pop_size, crossover_rate)

        # Try to mutate each child
        for i in range(pop_size):
            population[i] = mutate(population[i], mutation_rate)

        # Get the mean error of the population and track it for each generation
        temp_mean = stats.mean(temp_tuple[1])
        mean_error.append(temp_mean)

        # Print the progress periodically
        if (generation % 100 == 0):
            print("GA{2}: Generation {0}, Mean Error: {1}".format(generation, temp_mean, process_id))

        # Move to the next generation
        generation += 1

    # print("Finished GA training at {0}".format(time.ctime(time.time())))
    return mean_error


# Test run if this file is ran on its own
if __name__ == '__main__':
    af_path = 'datasets\\converted\\airfoil.csv'
    df = pandas.read_csv(af_path, header=None)
    training_data = df.values.tolist()
    nn = MLP.MLP(len(training_data[0]) - 1, 1, 10, training_data)
    mean_error = train(nn, 2000, 100, 0.5, 0.1)

    plt.plot(mean_error, label='GA')
    plt.xlabel('Generation')
    plt.ylabel('Mean Squared Error')
    plt.yscale('log')
    plt.title('GA')
    plt.legend()
    plt.show()
