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
import multiprocessing.managers as mng
import time
import pandas
import csv


class GAProcess(multiprocessing.Process):
    def __init__(self, process_ID, dataset_name, training_data, num_inputs, max_gen, pop_size, crossover_rate,
                 mutation_rate):
        multiprocessing.Process.__init__(self)
        self.process_ID = process_ID
        self.name = "GA {0}".format(dataset_name)
        # self.results = results
        self.training_data = training_data
        self.num_inputs = num_inputs
        self.max_gen = max_gen
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def run(self):
        print("Process {0}: Starting {1} training at {2}".format(self.process_ID, self.name, time.ctime(time.time())))
        nn = MLP.MLP(self.num_inputs, 1, 10, self.training_data)
        # temp_list = self.results
        # self.results.append((self.name, GA.train(nn, self.max_gen, self.pop_size, self.crossover_rate, self.mutation_rate, self.process_ID)))
        result = GA.train(nn, self.max_gen, self.pop_size, self.crossover_rate, self.mutation_rate, self.process_ID)
        # print(result)
        print("Process {0}: Finished {1} training at {2}".format(self.process_ID, self.name, time.ctime(time.time())))
        with open('Results\\{0}.csv'.format(self.name), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(result)


class ESProcess(multiprocessing.Process):
    def __init__(self, process_ID, dataset_name, training_data, num_inputs, max_gen, pop_size, crossover_rate,
                 num_children):
        multiprocessing.Process.__init__(self)
        self.process_ID = process_ID
        self.name = "ES {0}".format(dataset_name)
        # self.results = results
        self.training_data = training_data
        self.num_inputs = num_inputs
        self.max_gen = max_gen
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.num_children = num_children

    def run(self):
        print("Process {0}: Starting {1} training at {2}".format(self.process_ID, self.name, time.ctime(time.time())))
        nn = MLP.MLP(self.num_inputs, 1, 10, self.training_data)
        # self.results.put((self.name, ES.train(nn, self.max_gen, self.pop_size, self.num_children, self.crossover_rate, self.process_ID)))
        result = ES.train(nn, self.max_gen, self.pop_size, self.num_children, self.crossover_rate, self.process_ID)
        print("Process {0}: Finished {1} training at {2}".format(self.process_ID, self.name, time.ctime(time.time())))
        with open('Results\\{0}.csv'.format(self.name), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(result)


class DEProcess(multiprocessing.Process):
    def __init__(self, process_ID, dataset_name, training_data, num_inputs, max_gen, pop_size, crossover_rate, beta):
        multiprocessing.Process.__init__(self)
        self.process_ID = process_ID
        self.name = "DE {0}".format(dataset_name)
        # self.results = results
        self.training_data = training_data
        self.num_inputs = num_inputs
        self.max_gen = max_gen
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.beta = beta

    def run(self):
        print("Process {0}: Starting {1} training at {2}".format(self.process_ID, self.name, time.ctime(time.time())))
        nn = MLP.MLP(self.num_inputs, 1, 10, self.training_data)
        # self.results.put((self.name, DE.train(nn, self.max_gen, self.pop_size, self.crossover_rate, self.beta, self.process_ID)))
        result = DE.train(nn, self.max_gen, self.pop_size, self.crossover_rate, self.beta, self.process_ID)
        print("Process {0}: Finished {1} training at {2}".format(self.process_ID, self.name, time.ctime(time.time())))
        with open('Results\\{0}.csv'.format(self.name), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(result)


class BPProcess(multiprocessing.Process):
    def __init__(self, process_ID, dataset_name, training_data, num_inputs, iterations):
        multiprocessing.Process.__init__(self)
        self.process_ID = process_ID
        self.name = "BP {0}".format(dataset_name)
        # self.results = results
        self.training_data = training_data
        self.num_inputs = num_inputs
        self.iterations = iterations

    def run(self):
        print("Process {0}: Starting {1} training at {2}".format(self.process_ID, self.name, time.ctime(time.time())))
        nn = MLP.MLP(self.num_inputs, 1, 10, self.training_data)
        # self.results.put((self.name, nn.train(self.iterations)))
        result = nn.train(self.iterations, self.process_ID)
        print("Process {0}: Finished {1} training at {2}".format(self.process_ID, self.name, time.ctime(time.time())))
        with open('Results\\{0}.csv'.format(self.name), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(result)


def perform_experiment():
    csv_names = ['airfoil', 'concrete', 'forestfires', 'machine', 'yacht']
    # csv_names = ['airfoil']
    datasets = {}

    for i in range(len(csv_names)):
        datasets[csv_names[i]] = get_dataset('datasets\\converted\\{0}.csv'.format(csv_names[i]))

    max_gen = 10000
    pop_size = 100
    crossover_rate = 0.5
    mutation_rate = 0.1
    num_children = 100
    beta = 0.1
    max_iter = 100000
    processes = []
    process_counter = 0
    # manager = mng.SyncManager()
    # manager.start()
    # results = manager.list()

    for i in range(len(csv_names)):
        # results.append(manager.list())
        num_inputs = len(datasets[csv_names[i]][0]) - 1
        processes.append(GAProcess(process_counter, csv_names[i], datasets[csv_names[i]], num_inputs, max_gen, pop_size,
                                   crossover_rate, mutation_rate))
        processes[process_counter].start()
        process_counter += 1

        processes.append(ESProcess(process_counter, csv_names[i], datasets[csv_names[i]], num_inputs, max_gen, pop_size,
                                   crossover_rate, num_children))
        processes[process_counter].start()
        process_counter += 1

        processes.append(DEProcess(process_counter, csv_names[i], datasets[csv_names[i]], num_inputs, max_gen, pop_size,
                                   crossover_rate, beta))
        processes[process_counter].start()
        process_counter += 1

        processes.append(BPProcess(process_counter, csv_names[i], datasets[csv_names[i]], num_inputs, max_iter))
        processes[process_counter].start()
        process_counter += 1

    '''for i in range(len(results)):
        temp_results.append([])
        for j in range(4):
            temp_results[i].append(results[i].get())

    plt.plot(results)
    plt.xlabel('Generation')
    plt.ylabel('Mean Squared Error')
    plt.yscale('log')
    plt.title('GA')
    plt.legend()
    #plt.savefig('Test.png')
    plt.show()'''

    # print(results[1])
    # manager.join()


def print_results():
    pass


def get_dataset(csv_path):
    df = pandas.read_csv(csv_path, header=None)
    return df.values.tolist()


def main():
    num_process = 4
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
    plt.show()


if __name__ == '__main__':
    perform_experiment()
