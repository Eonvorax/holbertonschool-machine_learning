#!/usr/bin/env python3

"""
This is the DeepNeuralNetwork class module.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """
    Represents a deep neural network to perform classification.
    """

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network object.

        Args:
            nx (int): The number of input features.
            layers (list): A list of positive integers representing the
                number of nodes in each layer.

        Raises:
            TypeError: If nx is not an integer or layers is not a list of
                positive integers.
            ValueError: If nx is not a positive integer.

        Attributes:
            L (int): The number of layers in the neural network.
            cache (dict): A dictionary to hold intermediate values.
            weights (dict): A dictionary to hold the weights and biases.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if i == 0:
                # On layer 0: first layer's length is nx
                prev_layer = nx
            else:
                # Otherwise we use the previous layer length
                prev_layer = layers[i - 1]

            # He Normal (He-et-al) initialization
            self.__weights[f"W{i + 1}"] = np.random.randn(
                layers[i], prev_layer) * np.sqrt(2 / prev_layer)

            self.__weights[f"b{i + 1}"] = np.zeros((layers[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.
        """
        # Setting up the first input of first layer : X
        self.__cache["A0"] = X

        for i in range(1, self.__L + 1):
            # Previous layer activation ouput, used as input
            prev_A = self.__cache[f"A{i - 1}"]

            Z = np.matmul(self.__weights[f"W{i}"], prev_A)\
                + self.__weights[f"b{i}"]

            # Apply sigmoid for all layers except the last one
            if i < self.__L:
                activation = 1 / (1 + np.exp(-Z))
            else:
                # For the output layer, use softmax activation
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                activation = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

            # Store activation in cache
            self.__cache[f"A{i}"] = activation

        # The output of the network is in A{layer} (the last output)
        return self.__cache[f"A{self.__L}"], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        Y is a a one-hot numpy.ndarray of shape (classes, m) encoding the
        correct labels
        A is a numpy.ndarray with shape (1, m) containing the activated
            output of the network for each example.
        """
        m = Y.shape[1]
        return -(1 / m) * np.sum(Y * np.log(A))

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.

        X is a numpy.ndarray with shape (nx, m) that contains the input data.
        nx is the number of input features to the neuron.
        m is the number of examples.
        Y is a numpy.ndarray with shape (classes, m) that contains the correct
        one-hot encoded labels for the input data.

        Returns the NN's prediction (a numpy.ndarray with shape (1, m))
        and the cost of the network, respectively.
        """
        output, cache = self.forward_prop(X)

        # Similar trick as in one_hot_encode()
        prediction = np.eye(output.shape[0])[np.argmax(output, axis=0)].T

        # Cost of activated output on "output" layer
        cost = self.cost(Y, output)

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the network.

        Y is a numpy.ndarray with shape (classes, m) that contains the correct
        one-hot encoded labels for the input data.
        cache is a dictionary containing all the intermediary values
        alpha is the learning rate

        Updates the private attribute __weights
        """

        m = Y.shape[1]
        dZ = cache[f"A{self.__L}"] - Y

        # In reverse layer order :
        for i in range(self.__L, 0, -1):
            # Previous layer activation output
            prev_A = cache[f"A{i - 1}"]

            dW = (1 / m) * np.matmul(dZ, prev_A.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            # Prepare the next layer's gradient calculation
            dZ = np.matmul(
                    self.__weights[f"W{i}"].T, dZ) * prev_A * (1 - prev_A)

            # Updating parameters using the gradients and the learning rate
            self.__weights[f"W{i}"] -= alpha * dW
            self.__weights[f"b{i}"] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
            Method to train deep neural network

            :param X: ndarray, shape(nx,m), input data
            :param Y: ndarray, shape (classes, m) one-hot encoded labels
                for the correct output data.
            :param iterations: number of iterations to train over
            :param alpha: learning rate
            :param verbose: boolean print or not information
            :param graph: boolean print or not graph
            :param step: int

            :return: evaluation of training after iterations
        """

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean")
        if not isinstance(graph, bool):
            raise TypeError("graph must be a boolean")
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        # list to store cost /iter
        costs = []
        count = []

        for i in range(iterations + 1):
            # run forward propagation
            A, cache = self.forward_prop(X)

            # run gradient descent for all iterations except the last one
            if i != iterations:
                self.gradient_descent(Y, self.cache, alpha)

            cost = self.cost(Y, A)

            # store cost for graph
            costs.append(cost)
            count.append(i)

            # verbose TRUE, every step + first and last iteration
            if verbose and (i % step == 0 or i == 0 or i == iterations):
                # run evaluate
                print("Cost after {} iterations: {}".format(i, cost))

        # graph TRUE after training complete
        if graph:
            plt.plot(count, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format.
        filename is the file to which the object will be saved.
        If filename does not have the extension .pkl, adds it.
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object.
        filename is the file from which the object should be loaded.
        Returns: the loaded object, or None if filename doesn't exist.
        """

        try:
            with open(filename, "rb") as file:
                unpickled_obj = pickle.load(file)
            return unpickled_obj

        except FileNotFoundError:
            return None
