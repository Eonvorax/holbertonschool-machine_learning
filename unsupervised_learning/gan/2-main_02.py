#!/usr/bin/env python3

from math import sqrt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random

Simple_GAN = __import__('0-simple_gan').Simple_GAN
WGAN_clip = __import__('1-wgan_clip').WGAN_clip
WGAN_GP = __import__('2-wgan_gp').WGAN_GP

N = 49
sigma = .05
shape = [N, 2*N, 2*N, N]
lr = .00003
S = 500

# Regulating the seed


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def spheric_generator(nb_points, dim):
    u = tf.random.normal(shape=(nb_points, dim))
    return u/tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(u), axis=[1])+10**-8), [nb_points, 1])


def normal_generator(nb_points, dim):
    return tf.random.normal(shape=(nb_points, dim))


def uniform_generator(nb_points, dim):
    return tf.random.uniform(shape=(nb_points, dim))


# Code taken from above examples
def cloud_in_the_corner_numpy(N, sigma, S):
    arr = np.random.randn(S*N).reshape(S, N)
    return arr*sigma+.8


def cloud_in_the_corner(C, sigma, S):
    return tf.convert_to_tensor(cloud_in_the_corner_numpy(C, sigma, S), dtype="float32")


def mean_squared_dist_cloud_to_center(cloud):
    return np.mean((np.sum(np.square(cloud-.8), axis=1)))


def expected_mean_squared_dist_cloud_to_center(N, sigma, S):
    return sigma*sigma*N


def squared_dist_barycenter_to_center(cloud):
    S = cloud.shape[0]
    N = cloud.shape[1]
    barycenter = 1/S*np.sum(cloud, axis=0)
    diff = barycenter - .8
    return np.sum(np.square(diff))


def expected_squared_dist_barycenter_to_center(N, sigma, S):
    return sigma*sigma*N/S


def verif_squared_dist_barycenter_to_center(N, sigma, S):
    squared_dists = []
    for i in range(10000):
        cloud = cloud_in_the_corner_numpy(N, sigma, S)
        squared_dists.append(squared_dist_barycenter_to_center(cloud))
    print(f"""Among 10000 clouds of {S} points, the distribution of the squared distance
          from the barycenter of the cloud to the center has the following shape :""")
    a = plt.hist(squared_dists, bins=50, density=True)
    expected = expected_squared_dist_barycenter_to_center(N, sigma, S)
    plt.plot([expected, expected], [0, np.max(a[0])])
    plt.text(expected, np.max(a[0])*1.1,
             r'$\frac{N\sigma^2}{S}$', dict(size=10))
    plt.ylim(0, np.max(a[0])*1.2)
    plt.show()

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

real = cloud_in_the_corner(N, sigma, 10000)


# LET'S GO !

set_seeds(0)
Gs = [example_fully_connected_GAN(chosen_type, real, shape, 100, learning_rate=lr)
      for chosen_type in [Simple_GAN, WGAN_clip, WGAN_GP]]

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
names = ["Simple_GAN", "WGAN_clip", "WGAN_GP"]
for G, i, name in zip(Gs, range(3), names):
    axes[i].plot(G.history.history["discr_loss"])
    axes[i].plot(G.history.history["gen_loss"])
    axes[i].set_title(name)
    cloud = G.get_fake_sample(S).numpy()
    fmsdcc = mean_squared_dist_cloud_to_center(cloud)
    fdbc = squared_dist_barycenter_to_center(cloud)
    expected_msdcc = expected_mean_squared_dist_cloud_to_center(N, sigma, S)
    expected_dbc = expected_squared_dist_barycenter_to_center(N, sigma, S)
    axes[i].set_xlabel("msdcc : {:.4f} , dbc : {:.4f}".format(fmsdcc, fdbc))
fig.suptitle("expected msdcc : {:.4f} , expected dbc : {:.4f}".format(
    expected_msdcc, expected_dbc))
plt.tight_layout()
plt.savefig("tests.png")
plt.show()
