#!/usr/bin/env python3
"""
Wasserstein GAN with gradient penalty
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_GP(keras.Model):
    """
    This class represents a Wasserstein GAN (WGAN) with Gradient Penalty.
    """
    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005, lambda_gp=10):
        """
        Initializes the WGAN-GP model with a generator, discriminator, latent
        generator, real examples, and other parameters.
        """
        # run the __init__ of keras.Model first.
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        # standard value, but can be changed if necessary
        self.beta_1 = .3
        # standard value, but can be changed if necessary
        self.beta_2 = .9

        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1,
                             dtype='int32')
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(
            self.scal_shape)

        # define the generator loss and optimizer:
        self.generator.loss = lambda x: -tf.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(
            optimizer=generator.optimizer, loss=generator.loss)

        # define the discriminator loss and optimizer:
        self.discriminator.loss = lambda x, y: \
            tf.reduce_mean(y) - tf.reduce_mean(x)
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.discriminator.compile(
            optimizer=discriminator.optimizer, loss=discriminator.loss)

    # generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        """
        Generates a batch of fake samples using the generator.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        """
        Retrieves a batch of real samples from the dataset.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # generator of interpolating samples of size batch_size
    def get_interpolated_sample(self, real_sample, fake_sample):
        """
        Generates interpolated samples between real and fake samples.
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape)-u
        return u*real_sample+v*fake_sample

    # computing the gradient penalty
    def gradient_penalty(self, interpolated_sample):
        """
        Computes the gradient penalty for the interpolated samples.
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    # overloading train_step()
    def train_step(self, useless_argument):
        """
        Trains the GAN's discriminator and generator and returns their
        respective losses and gradient penalty.
        """
        for _ in range(self.disc_iter):
            # Tape watching the weights of the discriminator
            with tf.GradientTape() as disc_tape:
                # Get a real sample batch
                real_samples = self.get_real_sample()
                # Get a fake sample batch
                fake_samples = self.get_fake_sample(training=True)

                interpolated_sample = self.get_interpolated_sample(
                    real_samples, fake_samples)
                # Compute discriminator loss on both real and fake samples
                real_preds = self.discriminator(real_samples, training=True)
                fake_preds = self.discriminator(fake_samples, training=True)
                discr_loss = self.discriminator.loss(real_preds, fake_preds)

                gp = self.gradient_penalty(interpolated_sample)
                new_discr_loss = discr_loss + self.lambda_gp * gp
            # Compute the gradients for the discriminator
            discr_gradients = disc_tape.gradient(
                new_discr_loss, self.discriminator.trainable_variables)

            # Apply gradients to update discriminator weights
            self.discriminator.optimizer.apply_gradients(
                zip(discr_gradients, self.discriminator.trainable_variables))

        # Tape watching the weights of the discriminator
        with tf.GradientTape() as gen_tape:
            # Get a fake sample batch
            fake_samples = self.get_fake_sample(training=True)

            # Compute generator loss (how well the discriminator is fooled)
            fake_preds = self.discriminator(fake_samples, training=False)
            gen_loss = self.generator.loss(fake_preds)

        # Compute the gradients for the generator
        gen_gradients = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)

        # Apply gradients to update generator weights
        self.generator.optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables))

        # Return losses for both the discriminator and generator
        return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}
