#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random

convolutional_GenDiscr = __import__('3-generate_faces').convolutional_GenDiscr
Simple_GAN = __import__('0-simple_gan').Simple_GAN

# Regulating the seed


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def spheric_generator(nb_points, dim):
    u = tf.random.normal(shape=(nb_points, dim))
    return u/tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(u), axis=[1])+10**-8), [nb_points, 1])

# Load the pictures from small_res_faces_10000


array_of_pictures = np.load("small_res_faces_10000.npy")
array_of_pictures = array_of_pictures.astype("float32")/255

# Center and Normalize the data

mean_face = array_of_pictures.mean(axis=0)
centered_array = array_of_pictures - mean_face
multiplier = np.max(np.abs(array_of_pictures), axis=0)
normalized_array = centered_array/multiplier

# An example of a large real sample


def lat_gen(k): return spheric_generator(k, 16)*.3+.5


real_ex = tf.convert_to_tensor(normalized_array, dtype="float32")

# Visualize the faces


def recover(normalized):
    return normalized*multiplier+mean_face


def visualize_faces(G, lat_vecs, show_history=True):
    fig = plt.figure(figsize=(20, 4))
    wax = 4
    gs = gridspec.GridSpec(wax * 2, wax * 10)
    axes1 = [fig.add_subplot(gs[:wax - 1, i * wax:(i + 1) * wax - 1])
             for i in range(10)]
    axes2 = [fig.add_subplot(gs[wax:, 0:3 * wax]), fig.add_subplot(gs[wax:, 3 * wax + 1:6 * wax + 1]),
             fig.add_subplot(gs[wax:, 6 * wax + 2:9 * wax + 3])]

    generated_images = recover(G.generator(lat_vecs))

    for i in range(10):
        axes1[i].axis("off")
        axes1[i].imshow(generated_images[i, :, :, 0])

    b = G.discriminator(G.get_fake_sample(400)).numpy()
    a = G.discriminator(G.get_real_sample(400)).numpy()
    axes2[0].hist(a, bins=20)
    axes2[0].set_title("values of disc on real")
    axes2[1].hist(b, bins=20)
    axes2[1].set_title("values of disc on fake")
    if show_history:
        axes2[2].plot(G.history.history["discr_loss"])
        axes2[2].plot(G.history.history["gen_loss"])
        axes2[2].set_title("losses")

    plt.show()

# LET'S GO !


# Train
set_seeds(0)
gen, discr = convolutional_GenDiscr()
G = Simple_GAN(gen, discr, lat_gen, real_ex, disc_iter=1,
               batch_size=10, learning_rate=.001)
G.compile()
G.fit(real_ex, epochs=5, steps_per_epoch=5)

# Visualize
lat_vecs = lat_gen(10)
visualize_faces(G, lat_vecs)

print(gen.summary(line_length=100))
print(discr.summary(line_length = 100))
