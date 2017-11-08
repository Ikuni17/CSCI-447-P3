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

def evaluate():
	pass

def crossover_multipoint(parents, num_children):
	'''takes a list of parents and a desired number of children'''

	global crossover_rate

	children = []
	parent_num = 0

	# generate num_children children
	for index in range(num_children):
		print('generating child ' + str(index))
		children.append([])
		# decide each attribute for the current child
		for attribute in range(len(parents[0]) - 1):
			# alternates betwewn parent 1 and 2
			if random.random() < crossover_rate:
				# randomly selects a parent from parents
				parent_num = int(random.random() * len(parents))
			children[index].append(parents[parent_num][attribute])
		print(children[index])


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

		print('Points: ' + str(point_1) + ', ' + str(point_2))
		print(str(flatten(offspring_1)))
		print(str(flatten(offspring_2)))

	else:
		# crossover does not occur
		print('Crossover did not occur')

def flatten(input):
	return [item for sublist in input for item in sublist]

def mutate(child):
	global mutation_rate

	for attribute in range(len(child)):
		if random.random() < mutation_rate:
			print('mutation occured')
			# mutates an attribute by at most \pm 50%
			child[attribute] += (random.random() - 0.5) * child[attribute]
		else:
			print('mutation did not occur')
	print(str(child))

def selection(population):
	pass

def train():
	generation = 0

if __name__ == '__main__':
	# for testing
	for i in range(24):
		p1.append(0)
		p2.append(88)

	p = [p1, p2]

	crossover_multipoint(p, 10)
