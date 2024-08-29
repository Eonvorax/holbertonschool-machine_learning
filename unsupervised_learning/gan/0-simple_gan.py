#!/usr/bin/env python3
"""
Simple GAN
"""


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class Simple_GAN(keras.Model):
    """
    This class represents a simple GAN model
    """
    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """
        Constructor for the simple GAN model.

        :Parameters:
        - generator : `keras.Model`
            The generator model of the GAN, responsible for generating fake
            samples.
        - discriminator : `keras.Model`
            The discriminator model of the GAN, responsible for
            distinguishing between real and fake samples.
        - latent_generator : `function`
            A function that generates random latent vectors as input for the
            generator.
        - real_examples : `tf.Tensor`
            A tensor containing real samples from the dataset.
        - batch_size : `int (optional)`
            The size of the batches used for training. Default is 200.
        - disc_iter : `int (optional)`
            The number of times the discriminator is updated for each generator
            update. Default is 2.
        - learning_rate : `float (optional)`
            The learning rate for the Adam optimizer. Default is 0.005.
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
        self.beta_1 = .5
        # standard value, but can be changed if necessary
        self.beta_2 = .9

        # define the generator loss and optimizer:
        self.generator.loss = lambda x: tf.keras.losses.MeanSquaredError()(
            x, tf.ones(x.shape))
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(
            optimizer=generator.optimizer, loss=generator.loss)

        # define the discriminator loss and optimizer:
        self.discriminator.loss = lambda x, y: \
            tf.keras.losses.MeanSquaredError()(
                x, tf.ones(x.shape)) + tf.keras.losses.MeanSquaredError()(
                    y, -1 * tf.ones(y.shape))
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

    # overloading train_step()
    def train_step(self, useless_argument):
        """
        Trains the GAN's discriminator and generator and returns their
        respective losses. This method overloads the original keras.Model
        train_step method.

        :Returns:
        - A dictionary containing the discriminator and generator's
        respective losses:
            `{"discr_loss": discr_loss, "gen_loss": gen_loss}`
        """

        # Step 1: Train the discriminator multiple times (disc_iter)
        for _ in range(self.disc_iter):
            # Get a real sample batch
            real_samples = self.get_real_sample()

            # Get a fake sample batch
            fake_samples = self.get_fake_sample(training=True)

            # Tape watching the weights of the discriminator
            with tf.GradientTape() as disc_tape:
                # Compute discriminator loss on both real and fake samples
                real_preds = self.discriminator(real_samples)
                fake_preds = self.discriminator(fake_samples)
                discr_loss = self.discriminator.loss(real_preds, fake_preds)

            # Compute the gradients for the discriminator
            discr_gradients = disc_tape.gradient(
                discr_loss, self.discriminator.trainable_variables)

            # Apply gradients to update discriminator weights
            self.discriminator.optimizer.apply_gradients(
                zip(discr_gradients, self.discriminator.trainable_variables))

        # Step 2: Train the generator
        # Tape watching the weights of the discriminator
        with tf.GradientTape() as gen_tape:
            # Get a fake sample batch
            fake_samples = self.get_fake_sample(training=True)

            # Compute generator loss (how well the discriminator is fooled)
            fake_preds = self.discriminator(fake_samples)
            gen_loss = self.generator.loss(fake_preds)

        # Compute the gradients for the generator
        gen_gradients = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)

        # Apply gradients to update generator weights
        self.generator.optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables))

        # Return losses for both the discriminator and generator
        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
