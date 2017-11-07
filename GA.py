'''import itertools

thing = [[['A'], ['B'], ['C']],[['D'], ['E'], ['F']]]

thing = list(itertools.chain.from_iterable(thing))
thing = list(itertools.chain.from_iterable(thing))
print(thing)'''

import MLP
import random

crossover_rate = 1
mutation_rate = 0.1
evaluation = []

parent_1 = [] #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
parent_2 = [] #[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

def init_population(size):
	population = []
	weights = MLP.get_nn()
	for i in range(size):
		population.append(weights)
	return population

def evaluate():
	pass

def crossover(p1, p2):
	global crossover_rate
	offspring_1 = []
	offspring_2 = []

	# for testing
	for i in range(5):
		parent_1.append(0)
		parent_2.append(1)

	if random.random() < crossover_rate:
		# crossover occurs
		print('Crossover occured')

		# select crossover points
		point_1 = random.randrange(0, len(parent_1))
		point_2 = random.randrange(point_1, len(parent_1))

		offspring_1.append(parent_1[:point_1])
		offspring_1.append(parent_2[point_1:point_2])
		offspring_1.append(parent_1[point_2:])

		print('Points: ' + str(point_1) + ', ' + str(point_2))


		print(str(flatten(offspring_1)))
	else:
		# crossover does not occur
		print('Crossover did not occur')

def flatten(input):
	return [item for sublist in input for item in sublist]

def mutate(child):
	global mutation_rate
	pass

def selection(population):
	pass

def train():
	generation = 0

if __name__ == '__main__':
	crossover(parent_1, parent_2)
