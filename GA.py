'''import itertools

thing = [[['A'], ['B'], ['C']],[['D'], ['E'], ['F']]]

thing = list(itertools.chain.from_iterable(thing))
thing = list(itertools.chain.from_iterable(thing))
print(thing)'''

import MLP
import random

crossover_rate = 1
mutation_rate = 0.1
num_slice_points = 3
evaluation = []

parent_1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
parent_2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

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
	global num_slice_points

	pieces = [[]]
	# need to make sure these one is greater than the other
	point_1 = random.randrange(len(parent_1))
	point_2 = random.randrange(len(parent_2))

	if random.random() < crossover_rate:
		# crossover occurs
		print('Crossover occured')
		if num_slice_points > len(parent_1) - 1:
			print('invalid num_slice_points')
		else:
			for i in range(num_slice_points):
				# appends the first chunk to pieces

		print(str(pieces))
	else:
		# crossover does not occur
		print('Crossover did not occur')

def mutate(child):
	global mutation_rate
	pass

def selection(population):
	pass

def train():
	generation = 0

if __name__ == '__main__':
	crossover(parent_1, parent_2)
