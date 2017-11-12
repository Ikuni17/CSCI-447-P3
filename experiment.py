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

class GAProcess(multiprocessing.Process):
    def __init__(self, process_ID, dataset_name, training_data, num_inputs, results):
        multiprocessing.Process.__init__(self)
        self.process_ID = process_ID
        self.name = "GA{0}".format(self.process_ID)
        self.dataset_name = dataset_name
        self.training_data = training_data
        self.num_inputs = num_inputs
        self.results = results

    def run(self):
        print("Process {0}: Starting {1} training on {2} dataset at {3}".format(self.process_ID, self.name,
                                                                                self.dataset_name,
                                                                                time.ctime(time.time())))
        nn = MLP.MLP(self.num_inputs, 1, 10, self.training_data)
        self.results.put((self.name, GA.train(nn, 100000, 100, 0.5, 0.1, self.process_ID)))
        print("Process {0}: Finished {1} training on {2} dataset at {3}".format(self.process_ID, self.name,
                                                                                self.dataset_name,
                                                                                time.ctime(time.time())))


class ESProcess(multiprocessing.Process):
    def __init__(self, process_ID, dataset_name, training_data, num_inputs, results, num_children):
        multiprocessing.Process.__init__(self)
        self.process_ID = process_ID
        self.name = "ES{0}:{1}".format(self.process_ID, num_children)
        self.dataset_name = dataset_name
        self.training_data = training_data
        self.num_inputs = num_inputs
        self.results = results
        self.num_children = num_children

    def run(self):
        print("Process {0}: Starting {1} training on {2} dataset at {3}".format(self.process_ID, self.name,
                                                                                self.dataset_name,
                                                                                time.ctime(time.time())))
        nn = MLP.MLP(self.num_inputs, 1, 10, self.training_data)
        self.results.put((self.name, ES.train(nn, 100000, 100, self.num_children, 0.5, self.process_ID)))
        print("Process {0}: Finished {1} training on {2} dataset at {3}".format(self.process_ID, self.name,
                                                                                self.dataset_name,
                                                                                time.ctime(time.time())))


class DEProcess(multiprocessing.Process):
    def __init__(self, process_ID, dataset_name, training_data, num_inputs, results):
        multiprocessing.Process.__init__(self)
        self.process_ID = process_ID
        self.name = "DE{0}".format(self.process_ID)
        self.dataset_name = dataset_name
        self.training_data = training_data
        self.num_inputs = num_inputs
        self.results = results

    def run(self):
        print("Process {0}: Starting {1} training on {2} dataset at {3}".format(self.process_ID, self.name,
                                                                                self.dataset_name,
                                                                                time.ctime(time.time())))
        nn = MLP.MLP(self.num_inputs, 1, 10, self.training_data)
        self.results.put((self.name, DE.train(nn, 2000, 100, 0.5, 0.1, self.process_ID)))
        print("Process {0}: Finished {1} training on {2} dataset at {3}".format(self.process_ID, self.name,
                                                                                self.dataset_name,
                                                                                time.ctime(time.time())))


class BPProcess(multiprocessing.Process):
    def __init__(self, process_ID, dataset_name, training_data, num_inputs, results):
        multiprocessing.Process.__init__(self)
        self.process_ID = process_ID
        self.name = "BP{0}".format(self.process_ID)
        self.dataset_name = dataset_name
        self.training_data = training_data
        self.num_inputs = num_inputs
        self.results = results

    def run(self):
        print("Process {0}: Starting {1} training on {2} dataset at {3}".format(self.process_ID, self.name,
                                                                                self.dataset_name,
                                                                                time.ctime(time.time())))
        nn = MLP.MLP(self.num_inputs, 1, 10, self.training_data)
        self.results.put((self.name, nn.train(iterations=100000)))
        print("Process {0}: Finished {1} training on {2} dataset at {3}".format(self.process_ID, self.name,
                                                                                self.dataset_name,
                                                                                time.ctime(time.time())))


def perform_experiment():
    pass


def print_results():
    pass


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
    main()
