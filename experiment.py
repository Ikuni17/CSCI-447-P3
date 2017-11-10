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


def perform_experiment():
    pass


def print_results():
    pass


def main():
    num_inputs = 2
    training_data = rosen.generate(0, num_inputs)
    nn = MLP.MLP(num_inputs, 1, 100, training_data)


if __name__ == '__main__':
    main()
