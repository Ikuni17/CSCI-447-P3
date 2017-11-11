'''
CSCI 447: Project 3
Group 28: Trent Baker, Logan Bonney, Bradley White
November 13, 2017
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
        self.results.put((self.name, GA.train(nn, 2000, 100, 0.5, 0.1, self.process_ID)))
        print("Process {0}: Finished {1} training on {2} dataset at {3}".format(self.process_ID, self.name,
                                                                                self.dataset_name,
                                                                                time.ctime(time.time())))


def perform_experiment():
    pass


def print_results():
    pass


def main():
    # nn1 = MLP.MLP(num_inputs, 1, 10, training_data)
    # nn2 = MLP.MLP(num_inputs, 1, 100, training_data)
    # nn3 = MLP.MLP(num_inputs, 1, 10, training_data)

    # mean_error1 = GA.train(nn1, 2000, 100, 0.5, 0.1)
    # mean_error2 = GA.train(nn2, 10000, 100, 0.5, 0.1)
    # mean_error3 = GA.train(nn3, 2000, 200, 0.5, 0.1)

    # plt.plot(mean_error1, label='100 Pop, 10 HN')
    # plt.plot(mean_error2, label='200 Pop, 100 HN')
    # plt.plot(mean_error3, label='200 Pop, 10 HN')
    # plt.xlabel('Generation')
    # plt.ylabel('Mean Squared Error')
    # plt.yscale('log')
    # plt.title('Genetic Algorithm')
    # plt.legend()
    # plt.show()

    num_process = 4
    results = multiprocessing.Queue()
    ga_processes = []
    process_counter = 0

    for i in range(num_process):
        num_inputs = 2
        training_data = rosen.generate(0, num_inputs)

        ga_processes.append(GAProcess(process_counter, 'Rosen', training_data, num_inputs, results))
        process_counter += 1
        ga_processes[i].start()

    for i in range(len(ga_processes)):
        result = results.get()
        plt.plot(result[1], label=str(result[0]))
        #ga_processes[i].join()

    plt.xlabel('Generation')
    plt.ylabel('Mean Squared Error')
    plt.yscale('log')
    plt.title('Genetic Algorithm')
    plt.legend()
    plt.savefig('GA.png')
    #plt.show(block=False)
    plt.show()

if __name__ == '__main__':
    main()
