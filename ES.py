import GA
import random
import math


def add_sigmas(pop_size, individual_size):
    sigmas = []

    for i in range(pop_size):
        sigmas.append([])
        for j in range(individual_size):
            sigmas[i].append(random.gauss(0, 1))

    return sigmas


# p.218 second bullet
def update_sigmas(sigmas):
    tau = (1 / math.sqrt(2 * math.sqrt(len(sigmas[0]))))
    tau_prime = (1 / math.sqrt(2 * len(sigmas[0])))

    for i in range(len(sigmas)):
        for j in range(len(sigmas[i])):
            sigmas[i][j] = sigmas[i][j] * math.exp((tau_prime * random.gauss(0, 1)) + (tau * random.gauss(0, 1)))

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
