#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
Simple_GAN = __import__('0-simple_gan').Simple_GAN

# Regulating the seed


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def hash_tensor(tensor):
    return np.sum(np.array([hash(x) % 2**30 for x in tensor.numpy().flatten()])) % 2**30


def spheric_generator(nb_points, dim):
    u = tf.random.normal(shape=(nb_points, dim))
    return u/tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(u), axis=[1])+10**-8), [nb_points, 1])


def normal_generator(nb_points, dim):
    return tf.random.normal(shape=(nb_points, dim))


def uniform_generator(nb_points, dim):
    return tf.random.uniform(shape=(nb_points, dim))


# Building Generator and Discriminator

def fully_connected_GenDiscr(gen_shape, real_examples, latent_type="normal"):

    #   Latent generator
    if latent_type == "uniform":
        def latent_generator(k): return uniform_generator(k, gen_shape[0])
    elif latent_type == "normal":
        def latent_generator(k): return normal_generator(k, gen_shape[0])
    elif latent_type == "spheric":
        def latent_generator(k): return spheric_generator(k, gen_shape[0])

    #   Generator
    inputs = keras.Input(shape=(gen_shape[0], ))
    hidden = keras.layers.Dense(gen_shape[1], activation='tanh')(inputs)
    for i in range(2, len(gen_shape)-1):
        hidden = keras.layers.Dense(gen_shape[i], activation='tanh')(hidden)
    outputs = keras.layers.Dense(gen_shape[-1], activation='sigmoid')(hidden)
    generator = keras.Model(inputs, outputs, name="generator")

    #   Discriminator
    inputs = keras.Input(shape=(gen_shape[-1], ))
    hidden = keras.layers.Dense(gen_shape[-2],   activation='tanh')(inputs)
    for i in range(2, len(gen_shape)-1):
        hidden = keras.layers.Dense(gen_shape[-1*i], activation='tanh')(hidden)
    outputs = keras.layers.Dense(1,              activation='tanh')(hidden)
    discriminator = keras.Model(inputs, outputs, name="discriminator")

    return generator, discriminator, latent_generator

# illustration:


def illustr_fully_connected_GenDiscr():
    generator, discriminator, latent_generator = fully_connected_GenDiscr([
                                                                          1, 100, 100, 2], None)
    print(generator.summary())
    print(discriminator.summary())


# Training a fully connected GAN (Simple GAN)

def example_fully_connected_GAN(chosen_type, real_examples, gen_shape, epochs,  batch_size=200, steps_per_epoch=250, latent_type="normal", learning_rate=.005):
    generator, discriminator, latent_generator = fully_connected_GenDiscr(
        gen_shape, real_examples, latent_type=latent_type)
    G = chosen_type(generator, discriminator, latent_generator,
                    real_examples, learning_rate=learning_rate)
    G.compile()
    G.fit(real_examples, epochs=epochs,
          steps_per_epoch=steps_per_epoch, verbose=1)
    return G


# An example of a large real sample

def circle_example(k): return spheric_generator(k, 2)*.3+.5

# Visualize 2D the result of G


def visualize_2D(G, show=True, title=None, filename=None, dpi=200):
    fig = plt.figure(figsize=(16, 4))
    wax = 6  # width of ax
    gs = gridspec.GridSpec(wax, wax * 3 + 2)
    axes = [fig.add_subplot(gs[:, :wax]),
            fig.add_subplot(gs[:, wax + 1:2 * wax + 2]),
            fig.add_subplot(gs[:, 2 * wax + 2:])]

    # draw a real and a fake sample on axis 0:
    axes[0].set_xlim(-.1, 1.1)
    axes[0].set_ylim(-.1, 1.1)

    X = G.get_real_sample(size=200).numpy()
    axes[0].scatter(X[:, 0], X[:, 1], s=1)

    X = G.get_fake_sample(size=200).numpy()
    axes[0].scatter(X[:, 0], X[:, 1], s=1)

    axes[0].set_title("real sample (blue) vs fake sample (orange)")

    # draw the values of the discriminator on [-1,1]x[-1,1] on axis 1:
    axes[1].set_xlim(-.1, 1.1)
    axes[1].set_ylim(-.1, 1.1)

    A = np.linspace(-.1, 1.1, 150)
    B = np.linspace(-.1, 1.1, 150)
    X, Y = np.meshgrid(A, B)
    U = tf.convert_to_tensor(np.vstack([X.ravel(), Y.ravel()]).T)

    X = G.discriminator(U, training=False).numpy()
    u = axes[1].pcolormesh(A, B, X[:, 0].reshape(
        [150, 150]), shading='gouraud')

    axes[1].set_title("values of the discriminator")

    def sci(x): return "{:.2E}".format(.99 * x * (x > 0) + 1.01 * x * (x < 0))
    cb = fig.colorbar(u, ax=axes[1])
    # cb.set_ticks([sci(np.min(X[:,0])), 0, sci(np.max(X[:,0]))])
    cb.ax.tick_params(labelsize=5)

    # draw the training history on axis 3:
    HGL = G.history.history["gen_loss"]
    HDL = G.history.history["discr_loss"]

    axes[2].plot(np.arange(1, len(HGL) + 1), HGL)
    axes[2].plot(np.arange(1, len(HGL) + 1), HDL)
    axes[2].set_title("generator loss (blue) and discriminator loss (orange)")

    # some more tuning:
    for a in axes:
        a.tick_params(axis='both', which='major', labelsize=5)
        a.tick_params(axis='both', which='minor', labelsize=5)

    if title:
        fig.suptitle(title)

    if filename:
        plt.savefig(filename, dpi=dpi)
        plt.close()

    if show:
        plt.show()

# LET'S GO !


set_seeds(0)
G = example_fully_connected_GAN(Simple_GAN, circle_example(
    1000), [1, 10, 10, 2], 16, steps_per_epoch=100, learning_rate=.001)
visualize_2D(G)
