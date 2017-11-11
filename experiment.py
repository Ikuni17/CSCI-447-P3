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


def perform_experiment():
    pass


def print_results():
    pass


def main():
    num_inputs = 2
    training_data = rosen.generate(0, num_inputs)
    #nn1 = MLP.MLP(num_inputs, 1, 10, training_data)
    nn2 = MLP.MLP(num_inputs, 1, 100, training_data)
    #nn3 = MLP.MLP(num_inputs, 1, 10, training_data)

    #mean_error1 = GA.train(nn1, 2000, 100, 0.5, 0.1)
    mean_error2 = GA.train(nn2, 10000, 100, 0.5, 0.1)
    #mean_error3 = GA.train(nn3, 2000, 200, 0.5, 0.1)

    #plt.plot(mean_error1, label='100 Pop, 10 HN')
    plt.plot(mean_error2, label='200 Pop, 100 HN')
    #plt.plot(mean_error3, label='200 Pop, 10 HN')
    plt.xlabel('Generation')
    plt.ylabel('Mean Squared Error')
    plt.yscale('log')
    plt.title('Genetic Algorithm')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
