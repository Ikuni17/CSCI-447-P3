import GA
import random

def add_sigmas(pop_size, individual_size):
    sigmas = []

    for i in range(pop_size):
        sigmas.append([])
        for j in range(individual_size):
            sigmas[i].append(random.gauss(1, 1))

    return sigmas

def train():
    generation = 0
    max_gen = 2000
    pop_size = 100
    population = GA.init_population(pop_size)
    sigmas = add_sigmas(pop_size, len(population[0]))
    heat_size = 5

if __name__ == '__main__':
    train()