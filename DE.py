import GA

BETA = 0.1
crossover_prob = 0.1

def mutate(population):
	trial_vectors = []
	for i in range(len(population)):
		while(i == diff1 or i == diff2 or diff1 == diff2):
			diff1 = random.randrange(0, len(population)-1)
			diff2 = random.randrange(0, len(population)-1)
		trail_vectors.append(population[i]+[x * BETA for x in (diff1 - diff2)])
	return trial_vectors

def crossover(trial_vector, parent):
	# Generate list of potential corssover points
	crossover_points = range(0, len(parents)0-1)
	random.shuffle(crossover_points)
	child = parent

	# Force crossover for at least one point
	point = crossover_points.pop(0)
	child[point] = trial_vector[point]
	crossover_points.pop(0)
	for j in range(len(parent)):
		if random.uniform() < crossover_prob and j != point:
			child[j] = trail_vector[crossover_points[j]]





