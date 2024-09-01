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


def plot_400(G):
    Y = G.get_fake_sample(400)
    Z = G.discriminator(Y)
    Znu = Z.numpy()
    Ynu = Y.numpy()
    inds = np.argsort(Znu[:, 0])
    H = Ynu[inds, :, :, 0]
    fig, axes = plt.subplots(20, 20, figsize=(20, 20))
    fig.subplots_adjust(top=0.95)
    fig.suptitle("fake faces")
    for i in range(400):
        axes[i//20, i % 20].imshow(recover(H[i, :, :]))
        axes[i//20, i % 20].axis("off")
    plt.show()

# LET'S GO


set_seeds(0)
generator, discriminator = convolutional_GenDiscr()
G = WGAN_GP(generator, discriminator, latent_generator, real_ex,
            batch_size=200, disc_iter=2, learning_rate=.001)
G.replace_weights("generator.h5", "discriminator.h5")
plot_400(G)
