#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Deep27 = __import__('27-deep_neural_network').DeepNeuralNetwork
Deep28 = __import__('28-deep_neural_network').DeepNeuralNetwork
one_hot_encode = __import__('24-one_hot_encode').one_hot_encode
one_hot_decode = __import__('25-one_hot_decode').one_hot_decode

lib = np.load('../data/MNIST.npz')
X_train_3D = lib['X_train']
Y_train = lib['Y_train']
X_valid_3D = lib['X_valid']
Y_valid = lib['Y_valid']
X_test_3D = lib['X_test']
Y_test = lib['Y_test']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1)).T
X_test = X_test_3D.reshape((X_test_3D.shape[0], -1)).T
Y_train_one_hot = one_hot_encode(Y_train, 10)
Y_valid_one_hot = one_hot_encode(Y_valid, 10)
Y_test_one_hot = one_hot_encode(Y_test, 10)

print('Sigmoid activation:')
deep27 = Deep27.load('27-output.pkl')
A_one_hot27, cost27 = deep27.evaluate(X_train, Y_train_one_hot)
A27 = one_hot_decode(A_one_hot27)
accuracy27 = np.sum(Y_train == A27) / Y_train.shape[0] * 100
print("Train cost:", cost27)
print("Train accuracy: {}%".format(accuracy27))
A_one_hot27, cost27 = deep27.evaluate(X_valid, Y_valid_one_hot)
A27 = one_hot_decode(A_one_hot27)
accuracy27 = np.sum(Y_valid == A27) / Y_valid.shape[0] * 100
print("Validation cost:", cost27)
print("Validation accuracy: {}%".format(accuracy27))
A_one_hot27, cost27 = deep27.evaluate(X_test, Y_test_one_hot)
A27 = one_hot_decode(A_one_hot27)
accuracy27 = np.sum(Y_test == A27) / Y_test.shape[0] * 100
print("Test cost:", cost27)
print("Test accuracy: {}%".format(accuracy27))

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_test_3D[i])
    plt.title(A27[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

print('\nTanh activaiton:')

deep28 = Deep28.load('28-saved.pkl')
A_one_hot28, cost28 = deep28.train(X_train, Y_train_one_hot, iterations=100,
                                   step=10, graph=False)
A28 = one_hot_decode(A_one_hot28)
accuracy28 = np.sum(Y_train == A28) / Y_train.shape[0] * 100
print("Train cost:", cost28)
print("Train accuracy: {}%".format(accuracy28))
A_one_hot28, cost28 = deep28.evaluate(X_valid, Y_valid_one_hot)
A28 = one_hot_decode(A_one_hot28)
accuracy28 = np.sum(Y_valid == A28) / Y_valid.shape[0] * 100
print("Validation cost:", cost28)
print("Validation accuracy: {}%".format(accuracy28))
A_one_hot28, cost28 = deep28.evaluate(X_test, Y_test_one_hot)
A28 = one_hot_decode(A_one_hot28)
accuracy28 = np.sum(Y_test == A28) / Y_test.shape[0] * 100
print("Test cost:", cost28)
print("Test accuracy: {}%".format(accuracy28))
deep28.save('28-output')

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_test_3D[i])
    plt.title(A28[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
