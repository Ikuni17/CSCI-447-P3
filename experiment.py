'''
CSCI 447: Project 3
Group 28: Trent Baker, Logan Bonney, Bradley White
November 13, 2017

Datasets Used:
airfoil: Instances: 1503, Attributes: 6, Outputs: 1 (last index)
concrete: Instances: 1030, Attributes: 9, Outputs: 1 (last index)
forestfires: Instances: 517, Attributes: 11, Outputs: 1 (last index)
machine: Instances: 209, Attributes: 7, Outputs: 1 (last index)
yacht: Instances: 308, Attributes: 7, Outputs: 1 (last index)
'''

import MLP
import GA
import ES
import DE
import rosen_generator as rosen
import matplotlib.pyplot as plt
import multiprocessing
import time
import pandas
import csv


# Class to start a GA in it's own process
class GAProcess(multiprocessing.Process):
    def __init__(self, process_ID, dataset_name, training_data, num_inputs, max_gen, pop_size, crossover_rate,
                 mutation_rate):
        multiprocessing.Process.__init__(self)
        self.process_ID = process_ID
        self.name = "GA {0}".format(dataset_name)
        self.training_data = training_data
        self.num_inputs = num_inputs
        self.max_gen = max_gen
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def run(self):
        print("Process {0}: Starting {1} training at {2}".format(self.process_ID, self.name, time.ctime(time.time())))
        # Create a neural network with (inputs, hidden layers, hidden_nodes, dataset)
        nn = MLP.MLP(self.num_inputs, 1, 100, self.training_data)
        # Train the network, the result is a vector of the mean squared error for each generation's population
        result = GA.train(nn, self.max_gen, self.pop_size, self.crossover_rate, self.mutation_rate, self.process_ID)
        print("Process {0}: Finished {1} training at {2}".format(self.process_ID, self.name, time.ctime(time.time())))
        # Write the results to a new CSV
        with open('Results\\{0}.csv'.format(self.name), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(result)


# ES process, very similar to a GA process, takes a different parameter num_children for mu + lambda ES
class ESProcess(multiprocessing.Process):
    def __init__(self, process_ID, dataset_name, training_data, num_inputs, max_gen, pop_size, crossover_rate,
                 num_children):
        multiprocessing.Process.__init__(self)
        self.process_ID = process_ID
        self.name = "ES {0}".format(dataset_name)
        self.training_data = training_data
        self.num_inputs = num_inputs
        self.max_gen = max_gen
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.num_children = num_children

    def run(self):
        print("Process {0}: Starting {1} training at {2}".format(self.process_ID, self.name, time.ctime(time.time())))
        nn = MLP.MLP(self.num_inputs, 1, 100, self.training_data)
        result = ES.train(nn, self.max_gen, self.pop_size, self.num_children, self.crossover_rate, self.process_ID)
        print("Process {0}: Finished {1} training at {2}".format(self.process_ID, self.name, time.ctime(time.time())))
        with open('Results\\{0}.csv'.format(self.name), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(result)


# DE process, similar to the previous two classes. Has parameters specific to DE
class DEProcess(multiprocessing.Process):
    def __init__(self, process_ID, dataset_name, training_data, num_inputs, max_gen, pop_size, crossover_rate, beta):
        multiprocessing.Process.__init__(self)
        self.process_ID = process_ID
        self.name = "DE {0}".format(dataset_name)
        self.training_data = training_data
        self.num_inputs = num_inputs
        self.max_gen = max_gen
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.beta = beta

    def run(self):
        print("Process {0}: Starting {1} training at {2}".format(self.process_ID, self.name, time.ctime(time.time())))
        nn = MLP.MLP(self.num_inputs, 1, 100, self.training_data)
        result = DE.train(nn, self.max_gen, self.pop_size, self.crossover_rate, self.beta, self.process_ID)
        print("Process {0}: Finished {1} training at {2}".format(self.process_ID, self.name, time.ctime(time.time())))
        with open('Results\\{0}.csv'.format(self.name), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(result)


# Class to start a backpropagation in it's own process
class BPProcess(multiprocessing.Process):
    def __init__(self, process_ID, dataset_name, training_data, num_inputs, iterations):
        multiprocessing.Process.__init__(self)
        self.process_ID = process_ID
        self.name = "BP {0}".format(dataset_name)
        self.training_data = training_data
        self.num_inputs = num_inputs
        self.iterations = iterations

    def run(self):
        print("Process {0}: Starting {1} training at {2}".format(self.process_ID, self.name, time.ctime(time.time())))
        nn = MLP.MLP(self.num_inputs, 1, 100, self.training_data, learning_rate=0.001)
        result = nn.train(self.iterations, self.process_ID)
        print("Process {0}: Finished {1} training at {2}".format(self.process_ID, self.name, time.ctime(time.time())))
        with open('Results\\{0}.csv'.format(self.name), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(result)


# Runs the complete experiment, with each algorithm and dataset within it's own process. Very CPU intensive.
def perform_experiment():
    # Dataset names
    csv_names = ['airfoil', 'concrete', 'forestfires', 'machine', 'yacht']
    # Dictionary with dataset name as the key and a 2D list of vectors as the value
    datasets = {}

    # Populate the dictionary using helper function
    for i in range(len(csv_names)):
        datasets[csv_names[i]] = get_dataset('datasets\\converted\\{0}.csv'.format(csv_names[i]))

    # Maximum number of generations for all Evolutionary Algorithms (EA): GA, ES, DE
    max_gen = 10000
    # Maximum population size for EA
    pop_size = 100
    # Crossover rate for all EA
    crossover_rate = 0.5
    # Mutation rate for GA
    mutation_rate = 0.1
    # Lambda for ES(mu + lambda)
    num_children = 100
    # Beta for DE
    beta = 0.1
    # Maximum number of iterations for backprop
    max_iter = 10000
    # List of all process objects
    processes = []
    # Number of processes, used for unique IDs
    process_counter = 0

    # Iterate through all datasets
    for i in range(len(csv_names)):
        # Get the number of inputs for this dataset so the Neural Net constructor can split the inputs and outputs
        num_inputs = len(datasets[csv_names[i]][0]) - 1
        # Setup a GA process and start it with the current dataset
        processes.append(GAProcess(process_counter, csv_names[i], datasets[csv_names[i]], num_inputs, max_gen, pop_size,
                                   crossover_rate, mutation_rate))
        processes[process_counter].start()
        process_counter += 1

        # Setup an ES process and start it with the current dataset
        processes.append(ESProcess(process_counter, csv_names[i], datasets[csv_names[i]], num_inputs, max_gen, pop_size,
                                   crossover_rate, num_children))
        processes[process_counter].start()
        process_counter += 1

        # Setup a DE process and start it with the current dataset
        processes.append(DEProcess(process_counter, csv_names[i], datasets[csv_names[i]], num_inputs, max_gen, pop_size,
                                   crossover_rate, beta))
        processes[process_counter].start()
        process_counter += 1

        # Setup a BP process and start it with the current dataset
        processes.append(BPProcess(process_counter, csv_names[i], datasets[csv_names[i]], num_inputs, max_iter))
        processes[process_counter].start()
        process_counter += 1


# Read in a csv dataset, convert all values to numbers, and return as a 2D list
def get_dataset(csv_path):
    df = pandas.read_csv(csv_path, header=None)
    return df.values.tolist()


def main():
    valid_response = False
    while not valid_response:
        print("1. Perform Experiment (WARNING CPU INTENSIVE)\n2. Choose Algorithm\n3. Run Submission Test\n")
        choice = input("Choose an option number > ")
    '''num_process = 4
    results = multiprocessing.Queue()
    ga_processes = []
    es_processes = []
    num_children = [100, 150, 200, 250]
    process_counter = 0

    for i in range(num_process):
        num_inputs = 2
        training_data = rosen.generate(0, num_inputs)

        # es_processes.append(ESProcess(process_counter, 'Rosen', training_data, num_inputs, results, num_children[i]))
        ga_processes.append(GAProcess(process_counter, 'Rosen', training_data, num_inputs, results))
        process_counter += 1
        ga_processes[i].start()
        # es_processes[i].start()

    for i in range(process_counter):
        result = results.get()
        plt.plot(result[1], label=str(result[0]))
        # ga_processes[i].join()

    plt.xlabel('Generation')
    plt.ylabel('Mean Squared Error')
    plt.yscale('log')
    plt.title('GA')
    plt.legend()
    plt.savefig('GA.png')
    # plt.show(block=False)
    plt.show()'''


if __name__ == '__main__':
    perform_experiment()
