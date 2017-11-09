'''import itertools

thing = [[['A'], ['B'], ['C']],[['D'], ['E'], ['F']]]

thing = list(itertools.chain.from_iterable(thing))
thing = list(itertools.chain.from_iterable(thing))
print(thing)'''

import MLP
import random

crossover_rate = .5
mutation_rate = .1
evaluation = []

p1 = [] #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
p2 = [] #[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

def init_population(size):
	population = []
	weights = MLP.get_nn()
	for i in range(size):
		population.append(weights)
	return population

def evaluate(individual):
	return sum(individual)

def crossover_multipoint(parents, num_children):
	'''takes a list of parents and produces (num_children) children with a random number of randomly selected slice points '''

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




def crossover_2point(parent_1, parent_2):
	''' takes two parents and produces two offspring with 2 randomly selected slice points '''

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

def flatten(input):
	''' flattens a ragged 2-d array into a 1-d array '''
	return [item for sublist in input for item in sublist]

def mutate(child):
	''' has a (mutation_rate) chance to change each attribute randomly by up to \pm 50% '''

	global mutation_rate

	for attribute in range(len(child)):
		if random.random() < mutation_rate:
			# mutates an attribute by at most \pm 50%
			if child[attribute] == 0:
				child[attribute] += (sum(child) / len(child))
			else:
				child[attribute] += (random.random() - 0.5) * child[attribute]
		else:
			pass
			# print('mutation did not occur')
	return child

def selection(population):
	pass

def tournament_selection(population, heat_size):
	# UNTESTED BECAUSE WE DONT HAVE EVALUATE
	''' selects (num_select) individuals from (population) and holds a tournament with (heat_size) heats '''

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
	generation = 0

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

def generate_random_individual(length):
	individual = []
	for i in range(length):
		individual.append(random.random())
	return individual

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

	population = tournament_selection(population, heat_size)

	for i in population:
		sum += i[0]

	print(str(original_sum) + '\n' + str(sum))


if __name__ == '__main__':
	test_select()
