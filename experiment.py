'''
CSCI 447: Project 3
Group 28: Trent Baker, Logan Bonney, Bradley White
November 13, 2017

Main script to run the Evolutionary Algorithms. Some functionality requires Python 3.6 and third party libraries.


Datasets Used:
airfoil: Instances: 1503, Attributes: 6, Outputs: 1 (last index)
concrete: Instances: 1030, Attributes: 9, Outputs: 1 (last index)
forestfires: Instances: 517, Attributes: 11, Outputs: 1 (last index)
machine: Instances: 209, Attributes: 7, Outputs: 1 (last index)
yacht: Instances: 308, Attributes: 7, Outputs: 1 (last index)
'''

import DE
import ES
import GA
import MLP
import csv
import matplotlib.pyplot as plt
import multiprocessing
import pandas
import time


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


# Returns a user's choice for dataset to test
def choose_dataset():
    valid_response = False
    print("Choose a dataset:")
    print("1. Airfoil\n2. Concrete\n3. Forest Fires\n4. Machine\n5. Yacht\n6. Exit")

    # Loop until a valid response is received
    while not valid_response:
        try:
            choice = int(input("> "))
        except(ValueError):
            print("Please enter a valid response")
            continue

        if choice == 1:
            valid_response = True
            return ('airfoil', get_dataset('datasets\\converted\\airfoil.csv'))

        elif choice == 2:
            valid_response = True
            return ('concrete', get_dataset('datasets\\converted\\concrete.csv'))

        elif choice == 3:
            valid_response = True
            return ('forestfires', get_dataset('datasets\\converted\\forestfires.csv'))

        elif choice == 4:
            valid_response = True
            return ('machine', get_dataset('datasets\\converted\\machine.csv'))

        elif choice == 5:
            valid_response = True
            return ('yacht', get_dataset('datasets\\converted\\yacht.csv'))

        elif choice == 6:
            valid_response = True
            break

        else:
            print("Please enter a valid response")


# Create a graph from a user's choosen algorithm
def plot_choice(filename):
    data = pandas.read_csv('Results\\' + filename + '.csv', header=None).T
    plt.plot(data)
    plt.title(filename + ' Dataset')
    plt.xlabel('Generation')
    plt.ylabel('Mean Squared Error')
    plt.yscale('log')
    plt.show()


# Allow the user to choose an algorithm to test
def choose_algorithm():
    # Maximum number of generations for all Evolutionary Algorithms (EA): GA, ES, DE
    max_gen = 1000
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

    valid_response = False
    print("Choose an algorithm to train with:")
    print("1. Genetic Algorithm\n2. Evolutionary Strategy\n3. Differential Evolution\n4. Backpropagation\n5. Exit")

    # Loop until a valid response is received
    while not valid_response:
        try:
            choice = int(input("> "))
        except(ValueError):
            print("Please enter a valid response")
            continue

        if choice == 1:
            valid_response = True
            # Get the chosen dataset
            temp = choose_dataset()
            print("Creating GA Process with {0} generations and {1} population size".format(max_gen, pop_size))
            # Create a new process
            ga_proccess = GAProcess(0, temp[0], temp[1], len(temp[1][0]) - 1, max_gen, pop_size, crossover_rate,
                                    mutation_rate)
            # Start the process, then wait for it join
            ga_proccess.start()
            ga_proccess.join()
            # Plot the results
            plot_choice('GA {0}'.format(temp[0]))

        # Same flow as choice 1
        elif choice == 2:
            valid_response = True
            temp = choose_dataset()
            print("Creating ES Process with {0} generations and {1} population size".format(max_gen, pop_size))
            es_process = ESProcess(0, temp[0], temp[1], len(temp[1][0]) - 1, max_gen, pop_size, crossover_rate,
                                   pop_size)
            es_process.start()
            es_process.join()
            plot_choice('ES {0}'.format(temp[0]))

        # Same flow as choice 1
        elif choice == 3:
            valid_response = True
            temp = choose_dataset()
            print("Creating DE Process with {0} generations and {1} population size".format(max_gen, pop_size))
            de_process = DEProcess(0, temp[0], temp[1], len(temp[1][0]) - 1, max_gen, pop_size, crossover_rate, beta)
            de_process.start()
            de_process.join()
            plot_choice('DE {0}'.format(temp[0]))

        # Same flow as choice 1
        elif choice == 4:
            valid_response = True
            temp = choose_dataset()
            print("Creating BP Process with {0} iterations".format(max_iter))
            bp_process = BPProcess(0, temp[0], temp[1], len(temp[1][0]) - 1, max_iter)
            bp_process.start()
            bp_process.join()
            plot_choice('BP {0}'.format(temp[0]))

        elif choice == 5:
            valid_response = True
            break

        else:
            print("Please enter a valid response")


# Run the test used for submission, which runs all algorithms concurrently on the same dataset and plots the results.
def submission_test():
    data = get_dataset('datasets\\converted\\machine.csv')
    algorithms = ['BP', 'DE', 'ES', 'GA']
    # Maximum number of generations for the Evolutionary Algorithms (EA)
    max_gen = 1000
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
    num_inputs = len(data[0]) - 1

    # Start each algorithm in its own process
    processes.append(
        GAProcess(process_counter, 'machine', data, num_inputs, max_gen, pop_size, crossover_rate, mutation_rate))
    processes[process_counter].start()
    process_counter += 1

    processes.append(
        ESProcess(process_counter, 'machine', data, num_inputs, max_gen, pop_size, crossover_rate, num_children))
    processes[process_counter].start()
    process_counter += 1

    processes.append(
        DEProcess(process_counter, 'machine', data, num_inputs, max_gen, pop_size, crossover_rate, beta))
    processes[process_counter].start()
    process_counter += 1

    processes.append(BPProcess(process_counter, 'machine', data, num_inputs, max_iter))
    processes[process_counter].start()
    process_counter += 1

    # Wait to join all the processes
    for process in processes:
        process.join()

    # Plot a large figure
    plt.figure(figsize=(25.5, 13.5), dpi=100)

    # Add all results to a plot
    for alg in algorithms:
        temp_df = pandas.read_csv('Results\\' + alg + ' machine.csv', header=None).T
        plt.plot(temp_df, label=alg)

    # Set the plot parameters and show it
    plt.title('Machine Dataset')
    plt.xlabel('Generation')
    plt.ylabel('Mean Squared Error')
    plt.yscale('log')
    plt.xlim(-100, 1000)
    plt.ylim(0, 100000)
    plt.legend()
    plt.show()


# Prompts the user to pick start the experiment, a single algorithm, or the submission test
def main():
    valid_response = False
    print("Choose an option:")
    print("1. Perform Experiment (WARNING CPU INTENSIVE)\n2. Choose Algorithm\n3. Run Submission Test\n4. Exit")

    # Loop until a valid response is received
    while not valid_response:
        try:
            choice = int(input("> "))
        except(ValueError):
            print("Please enter a valid response")
            continue

        if choice == 1:
            valid_response = True
            perform_experiment()

        elif choice == 2:
            valid_response = True
            choose_algorithm()

        elif choice == 3:
            valid_response = True
            submission_test()

        elif choice == 4:
            valid_response = True
            break

        else:
            print("Please enter a valid response")


if __name__ == '__main__':
    main()
