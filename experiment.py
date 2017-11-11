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
import threading
import time


class GAThread(threading.Thread):
    def __init__(self, thread_ID, dataset_name, training_data, num_inputs, results):
        threading.Thread.__init__(self)
        self.thread_ID = thread_ID
        self.name = "GA{0}".format(self.thread_ID)
        self.dataset_name = dataset_name
        self.training_data = training_data
        self.num_inputs = num_inputs
        self.results = results

    def run(self):
        print("Thread {0}: Starting {1} training on {2} dataset at {3}".format(self.thread_ID, self.name,
                                                                               self.dataset_name,
                                                                               time.ctime(time.time())))
        nn = MLP.MLP(self.num_inputs, 1, 10, self.training_data)
        self.results[self.thread_ID] = GA.train(nn, 1000, 100, 0.5, 0.1)
        print("Thread {0}: Finished {1} training on {2} dataset at {3}".format(self.thread_ID, self.name,
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

    results = [None] * 10
    ga_threads = []
    thread_counter = 0

    for i in range(2):
        num_inputs = 2
        training_data = rosen.generate(0, num_inputs)

        ga_threads.append(GAThread(thread_counter, 'Rosen', training_data, num_inputs, results))
        thread_counter += 1
        ga_threads[i].start()

    for i in range(len(ga_threads)):
        ga_threads[i].join()

    for i in range(len(results)):
        if results[i] is not None:
            plt.plot(results[i], label=str(i))

    plt.xlabel('Generation')
    plt.ylabel('Mean Squared Error')
    plt.yscale('log')
    plt.title('Genetic Algorithm')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
