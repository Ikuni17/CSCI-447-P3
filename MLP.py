'''
CSCI 447: Project 3
Group 28: Trent Baker, Logan Bonney, Bradley White
November 13, 2017
'''

import random
import numpy as np
import rosen_generator as rosen
import itertools
import matplotlib.pyplot as plt
import pandas
import math


class MLP:
    def __init__(self, num_inputs, num_hidden_layers, nodes_per_layer, training_data, learning_rate=0.1):
        self.weights = []  # Each numpy array in this represents the weights coming into a node
        self.inputs = []
        self.train_in = []
        self.train_out = []
        self.activation = []  # Each numpy array in this represents the activation leaving a node for every input
        self.learning_rate = learning_rate
        for x in training_data:
            self.train_in.append(x[:num_inputs])
            self.train_out.append(x[num_inputs:])
        print('Setting up the network with {0} inputs and {1} output(s)'.format(len(self.train_in[0]),
                                                                                len(self.train_out[0])))
        # print('Train_in: {0}'.format(train_in))
        # Initialize the NN with random weights and populate the activation matrix
        for i in range(num_hidden_layers + 1):
            self.weights.append([])  # append a matrix to represent a layer in the NN
            if i == 0:
                # this transposes the input to the right format and adds as the first activation layer
                self.activation.append(np.array(self.train_in).transpose())
                num_nodes = nodes_per_layer
                num_weights = num_inputs
            # use the number of outputs for the last layer
            elif i == num_hidden_layers:
                num_weights = num_nodes
                num_nodes = len(self.train_out[0])
                self.activation.append(np.array([]))
            # Otherwise it's a hidden_layer
            else:
                num_weights = num_nodes
                num_nodes = nodes_per_layer
                self.activation.append(np.array([]))

            # Each matrix in a layer represents a node and holds all the weights coming into the node
            for j in range(num_nodes):
                temp = []
                for k in range(num_weights):
                    temp.append(random.uniform(0, 1000))
                self.weights[i].append(np.array(temp))
        self.activation.append(np.array([]))

    def swap_weights(self, weights):
        # Layers
        for i in range(len(self.weights)):
            # Nodes with weight arrays
            for j in range(len(self.weights[i])):
                self.weights[i][j] = weights[:self.activation[i]]
        self.feedforward()

    def calc_avg_error(self):
        return np.average(np.square(np.array(self.train_out).transpose() - self.activation[len(self.activation) - 1]))

    def train(self, iterations=1000, process_ID=0):
        error_vector = []
        large_iter = False

        if iterations > 50000:
            data_freq = math.ceil(iterations / 50000)
            large_iter = True

        for i in range(iterations):
            self.feedforward()
            self.backprop()

            if large_iter:
                if i % data_freq == 0:
                    temp_mean = self.calc_avg_error()
                    error_vector.append(temp_mean)
                    if i % 1000 == 0:
                        print('BP{2}: Iteration {0}, Mean Error:{1}'.format(i, temp_mean, process_ID))
            else:
                temp_mean = self.calc_avg_error()
                error_vector.append(temp_mean)
                if i % 1000 == 0:
                    print('BP{2}: Iteration {0}, Mean Error:{1}'.format(i, temp_mean, process_ID))

        return error_vector

    # updates the activation arrays and the output
    def feedforward(self):
        for i in range(len(self.weights)):
            temp = np.zeros((len(self.weights[i]), len(self.activation[0][0])))
            for j in range(len(self.weights[i])):
                # append all the activations to list to be converted to an np array as an activation layer
                temp[j] = self.activation[i].transpose().dot(self.weights[i][j])
            # don't run the activation on the tanh function
            if (i == len(self.weights) - 1):
                self.activation[i + 1] = np.array(temp)
            else:
                self.activation[i + 1] = np.tanh(temp)

    def backprop(self):
        # print('Expected: {0}, Actual: {1}'.format(self.train_out[0], self.activation[len(self.activation)-1]))
        errors = np.subtract(self.activation[len(self.activation) - 1], np.array(self.train_out).transpose())[0]
        # error = self.calc_avg_error()
        for i, layer in reversed(list(enumerate(self.weights))):
            for j in range(len(layer)):
                activ_out = self.activation[i + 1][j]
                activ_in = self.activation[i][0]
                update = 0
                for k in range(len(errors)):
                    if i == len(self.weights) - 1:
                        modifier = -errors[k] * activ_in[k]
                    else:
                        modifier = activ_in[k] * errors[k] * (1 - (activ_out[k] ** 2))
                    update = update - modifier
                self.weights[i][j] = self.weights[i][j] - self.learning_rate * update * (1 / len(self.train_out))

    def print_nn(self):
        print('Dimensions of weights \n --------------------------')
        for layers in self.weights:
            for weight in layers:
                print(weight.shape, end='')
            print('\n')
        print('Dimensions of activations \n --------------------------')
        for layer in self.activation:
            print(layer.shape)

    # Calculate the activations of a node given it's wieghts and the values coming into the node
    @staticmethod
    def activ_fun(activ, weights):
        x = np.array(activ).transpose()
        y = np.array(weights)
        z = np.tanh(y.dot(x))
        return z

    def get_weights(self):
        return list(itertools.chain.from_iterable(itertools.chain.from_iterable(self.weights)))

    def set_weights(self, new_weights):
        counter = 0

        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] = new_weights[counter]
                    counter += 1


def main():
    af_path = 'datasets\\converted\\airfoil.csv'
    df = pandas.read_csv(af_path, header=None)
    training_data = df.values.tolist()
    # num_inputs = 2
    # training_data = rosen.generate(0, num_inputs)
    mlp = MLP(len(training_data[0]) - 1, 1, 10, training_data)
    error_vector = mlp.train(iterations=1000)

    plt.plot(error_vector, label='BP')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.yscale('log')
    plt.title('BP')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
