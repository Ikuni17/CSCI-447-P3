import MLP
import rosen_generator as rosen
import time
import random
import statistics as stats

crossover_rate = .5
mutation_rate = .1
evaluation = []
pop_error = []
num_inputs = 2
training_data = rosen.generate(0, num_inputs)
mlp = MLP.MLP(num_inputs, 1, 100, training_data)


def init_population(size):
    global mlp

    population = []
    num_weights = len(mlp.get_weights())
    for i in range(size):
        population.append(generate_random_individual(num_weights))
    return population


def generate_random_individual(length):
    individual = []
    for i in range(length):
        individual.append(random.uniform(1, 10000))
    return individual


def evaluate(individual):
    global mlp

    # print(mlp.weights)
    # print(individual)
    # Populate the network with this individual weights
    mlp.set_weights(individual)
    # Forward propagate
    mlp.feedforward()

    # Return the error of this individual
    return mlp.calc_avg_error()


# Takes a list of parents and produces (num_children) children with a random number of randomly selected slice points
def crossover_multipoint(parents, num_children):
    global crossover_rate

    children = []
    parent_num = 0

    # generate num_children children
    for index in range(num_children):
        children.append([])
        # decide each attribute for the current child
        for attribute in range(len(parents[0])):
            if random.random() < crossover_rate:
                # randomly select a parent from parents
                parent_num = int(random.random() * len(parents))
            children[index].append(parents[parent_num][attribute])
    return children


# Takes two parents and produces two offspring with 2 randomly selected slice points
def crossover_2point(parent_1, parent_2):
    global crossover_rate
    offspring_1 = []
    offspring_2 = []

    if random.random() < crossover_rate:
        # crossover occurs
        print('Crossover occured')

        # select crossover points
        point_1 = random.randrange(0, len(parent_1))
        point_2 = random.randrange(point_1 + 1, len(parent_1))

        offspring_1.append(parent_1[:point_1])
        offspring_1.append(parent_2[point_1:point_2])
        offspring_1.append(parent_1[point_2:])

        offspring_2.append(parent_2[:point_1])
        offspring_2.append(parent_1[point_1:point_2])
        offspring_2.append(parent_2[point_2:])

    else:
        # crossover does not occur
        print('Crossover did not occur')


# Has a (mutation_rate) chance to change each attribute randomly by up to +/- 50%
def mutate(child):
    global mutation_rate

    for attribute in range(len(child)):
        if random.random() < mutation_rate:
            # mutates an attribute by at most +/- 50%
            if child[attribute] == 0:
                child[attribute] += (sum(child) / len(child))
            else:
                child[attribute] += (random.random() - 0.5) * child[attribute]
                # else:
                #    pass
                # print('mutation did not occur')
    return child


def rank_selection(population, pop_size):
    global pop_error
    pop_error = []

    rank_weights = []
    for individual in population:
        fitness = evaluate(individual)
        rank_weights.append(1 / fitness)
        pop_error.append(fitness)

    return random.choices(population, rank_weights, k=pop_size)


# UNTESTED BECAUSE WE DONT HAVE EVALUATE
# Selects (num_select) individuals from (population) and holds a tournament with (heat_size) heats
def tournament_selection(population, heat_size):
    num_select = len(population)
    selected = []

    # to select num_select individuals
    for i in range(num_select):
        # randomly select heat_size individuals from the population
        heat = []
        for individual in range(heat_size):
            # add a random individual to heat
            heat.append(population[int(random.random() * len(population))])

        # find the best individual from heat and add it to selected
        # ASSUMING MINIMIZATION
        if heat != []:
            min = evaluate(heat[0])
            min_index = 0

            for contestant in heat:
                temp_fitness = evaluate(contestant)
                if temp_fitness < min:
                    min = temp_fitness
                    min_index = heat.index(contestant)

            selected.append(heat[min_index])
    return selected


def train():
    global evaluation
    global mlp
    global pop_error

    generation = 0
    max_gen = 2000
    pop_size = 200
    population = init_population(pop_size)
    heat_size = 10

    # Calculate the fitness of the first individual and set all individual's fitness to it since they are all the
    # same initially, not being used currently
    # first_eval = evaluate(population[0])
    # evaluation = [first_eval] * pop_size

    print("Starting GA training at {0}".format(time.ctime(time.time())))

    # TODO stop when converged?
    while (generation < max_gen):
        # Select the best parents and use them to produce pop_size children and overwrite the entire population
        population = crossover_multipoint(rank_selection(population, len(population)), pop_size)
        # population = crossover_multipoint(tournament_selection(population, heat_size), pop_size)

        # Try to mutate each child
        for i in range(len(population)):
            population[i] = mutate(population[i])

        if (generation % 5 == 0):
            print("Generation {0}, Error: {1}".format(generation, stats.mean(pop_error)))
        # Move to the next generation
        generation += 1

    print("Finished GA training at {0}".format(time.ctime(time.time())))


if __name__ == '__main__':
    train()

'''
Legacy Code
p1 = []  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
p2 = []  # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# Flattens a ragged 2-d array into a 1-d array
def flatten(input):
    return [item for sublist in input for item in sublist]


def selection(population):
    pass

def test_cross_mutate():
    print('parameters')
    print('crossover_rate: ' + str(crossover_rate))
    print('mutation_rate: ' + str(mutation_rate))

    p1 = []
    p2 = []
    for i in range(10):
        p1.append(0)
        p2.append(1)

    parents = [p1, p2]

    print('\nparents')
    for parent in parents:
        print(str(parent))

    children = crossover_multipoint([p1, p2], 10)
    print('\nchildren')
    for individual in children:
        print(str(individual))

    for individual in children:
        individual = mutate(individual)

    print('\nmutated children')
    for individual in children:
        print(str(individual))


def test_select():
    population_size = 50
    heat_size = 5
    length = 4

    population = []

    for i in range(population_size):
        population.append(generate_random_individual(length))

    print('\ntotal')
    original_sum = 0
    sum = 0
    for i in population:
        original_sum += i[0]

    population = rank_selection(population)

    for i in population:
        sum += i[0]

    print(str(original_sum) + '\n' + str(sum))


'''
