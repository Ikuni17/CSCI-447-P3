'''
CSCI 447: Project 3
Group 28: Trent Baker, Logan Bonney, Bradley White
November 13, 2017

This file contain the functionality to create a neural network and train it with forward and backward propagation.
'''

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas
import random


# Class to represent a neural network
class MLP:
    def __init__(self, num_inputs, num_hidden_layers, nodes_per_layer, training_data, learning_rate=0.01):
        self.weights = []  # Each numpy array in this represents the weights coming into a node
        self.inputs = []
        self.train_in = []
        self.train_out = []
        self.activation = []  # Each numpy array in this represents the activation leaving a node for every input
        self.learning_rate = learning_rate

        # Slice the training data into inputs and outputs and store each in a matrix
        for x in training_data:
            self.train_in.append(x[:num_inputs])
            self.train_out.append(x[num_inputs:])

        # Convert to numpy arrays for linear algebra
        self.train_in = np.array(self.train_in).astype('float64')
        self.train_out = np.array(self.train_out).astype('float64')

        print('Setting up the network with {0} inputs, {1} output(s), and {2} training examples.'.format(
            len(self.train_in[0]), len(self.train_out[0]), len(self.train_out)))

        # Initialize the NN with random weights and populate the activation matrix
        for i in range(num_hidden_layers + 1):
            self.weights.append([])  # append a matrix to represent a layer in the NN
            if i == 0:
                # this transposes the input to the right format and adds as the first activation layer
                self.activation.append(self.train_in.transpose())
                self.activation[0] = self.activation[0].astype('float64')
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
                    if (k != num_weights - 1):
                        temp.append(random.uniform(0, 10))
                    else:
                        temp.append(random.uniform(0, 10))
                self.weights[i].append(np.array(temp))
        self.activation.append(np.array([]))

    # Calculate the average error for a set of weights for all inputs
    def calc_avg_error(self):
        return np.average(
            np.square(self.train_out.transpose().astype('float64') - self.activation[len(self.activation) - 1]))

    # Generate a random individual for Evolutionary Algorithms
    def generate_random_individual(self, length):
        individual = []
        for i in range(length):
            if (i < length - len(self.train_out[0]) * len(self.weights[0])):
                individual.append(random.uniform(0, 1))
            else:
                individual.append(random.uniform(0, 1))
        return individual

    # Train the neural network with forward and backward propagation
    def train(self, iterations=1000, process_ID=0, learning_rate=0.01):
        self.learning_rate = learning_rate
        error_vector = []

        for i in range(iterations):
            self.feedforward()
            self.backprop()

            # Calculate the mean for this iteration
            temp_mean = self.calc_avg_error()
            error_vector.append(temp_mean)

            # Periodically print the progress
            if i % 1000 == 0:
                print('BP{2}: Iteration {0}, Mean Error: {1}'.format(i, temp_mean, process_ID))

        return error_vector

    # Updates the activation arrays and the output for all inputs
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

    # Backpropagate through the neural network to update weights
    def backprop(self):
        # Calculate the error for the most recent output
        errors = np.subtract(self.activation[len(self.activation) - 1], self.train_out.transpose())[0]

        # Start from the output layer
        for i, layer in reversed(list(enumerate(self.weights))):
            # Iterate through all nodes
            for j in range(len(layer)):
                # Get the input and output for each node
                activ_out = self.activation[i + 1][j]
                activ_in = self.activation[i][0]
                update = 0

                # Calculate the modifier based on each outputs error
                for k in range(len(errors)):
                    # Check if we're at the output layer
                    if i == len(self.weights) - 1:
                        modifier = -errors[k] * activ_in[k]
                    else:
                        modifier = activ_in[k] * errors[k] * (1 - (activ_out[k] ** 2))
                    update = update - modifier
                # Update the weight after the modifier has been completely calculated
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

    # Flatten the weights into a 1D array
    def get_weights(self):
        return list(itertools.chain.from_iterable(itertools.chain.from_iterable(self.weights)))

    # Expand a 1D array into the matrix used for forward propagation
    def set_weights(self, new_weights):
        counter = 0

        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] = new_weights[counter]
                    counter += 1


# Used for testing this file on its own
def main():
    af_path = 'datasets\\converted\\machine.csv'
    df = pandas.read_csv(af_path, header=None)
    training_data = df.values.tolist()
    mlp = MLP(len(training_data[0]) - 1, 1, 10, training_data)
    error_vector = mlp.train(iterations=10000, learning_rate=0.1)
    mlp = MLP(len(training_data[0]) - 1, 1, 10, training_data)
    error_vector1 = mlp.train(iterations=10000, learning_rate=0.01)
    mlp = MLP(len(training_data[0]) - 1, 1, 10, training_data)
    error_vector2 = mlp.train(iterations=10000, learning_rate=0.001)
    mlp = MLP(len(training_data[0]) - 1, 1, 10, training_data)
    error_vector3 = mlp.train(iterations=10000, learning_rate=0.0001)

    plt.plot(error_vector, label='0.1')
    plt.plot(error_vector1, label='0.01')
    plt.plot(error_vector2, label='0.001')
    plt.plot(error_vector3, label='0.0001')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.yscale('log')
    plt.title('BP Learning Rate')
    plt.legend()
    plt.savefig('BP Learning Rate.png')
    plt.show()


if __name__ == "__main__":
    main()
