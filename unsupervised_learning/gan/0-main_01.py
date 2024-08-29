#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

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

# LET'S GO !


set_seeds(0)
G = example_fully_connected_GAN(Simple_GAN,circle_example(1000), [1,10,10,2], 16, steps_per_epoch=100, learning_rate=.001)
