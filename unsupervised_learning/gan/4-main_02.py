#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

WGAN_GP = __import__('4-wgan_gp').WGAN_GP
convolutional_GenDiscr = __import__('3-generate_faces').convolutional_GenDiscr


# Regulating the seed

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def normal_generator(nb_points, dim):
    return tf.random.normal(shape=(nb_points, dim))


# Load the pictures from small_res_faces_10000

array_of_pictures = np.load("small_res_faces_10000.npy")
array_of_pictures = array_of_pictures.astype("float32")/255

# Center and Normalize the data

mean_face = array_of_pictures.mean(axis=0)
centered_array = array_of_pictures - mean_face
multiplier = np.max(np.abs(array_of_pictures), axis=0)
normalized_array = centered_array/multiplier

# An example of a large real sample


def latent_generator(k): return normal_generator(k, 16)


real_ex = tf.convert_to_tensor(normalized_array, dtype="float32")

# Visualize the fake faces : Code to generate the first picture


def recover(normalized):
    return normalized*multiplier+mean_face


def segment(ab, n):
    x = np.linspace(0, 1, n)
    return (1-x)[:, None]*ab[0, :][None, :]+x[:, None]*ab[1, :][None, :]


def plot_segment(G, ab, n):
    lats = tf.convert_to_tensor(segment(ab, n))
    Y = G.generator(lats)[:, :, :, 0]
    fig, axes = plt.subplots(1, n, figsize=(n+1, 1))
    fig.subplots_adjust(top=0.8)
    fig.suptitle("segment")
    for i in range(n):
        axes[i].imshow(recover(Y[i, :, :]))
        axes[i].axis("off")
    plt.savefig("segment.png")
    plt.show()


# LET'S GO

set_seeds(18)
generator, discriminator = convolutional_GenDiscr()
G = WGAN_GP(generator, discriminator, latent_generator, real_ex,
            batch_size=200, disc_iter=2, learning_rate=.001)
G.replace_weights("generator.h5", "discriminator.h5")

ab = np.random.randn(32).reshape([2, 16])
plot_segment(G, ab, 20)
