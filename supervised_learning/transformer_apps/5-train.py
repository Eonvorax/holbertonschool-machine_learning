#!/usr/bin/env python3
"""
Training a full Transformer network
"""

import tensorflow as tf
import keras as K
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(K.optimizers.schedules.LearningRateSchedule):
    def __init__(self, dm, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)

def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    """
    data = Dataset(batch_size, max_len)
    data_train = data.data_train
    data_valid = data.data_valid

    model = Transformer(N, dm, h, hidden,
                        input_vocab=data.tokenizer_pt.vocab_size + 2,  # To account for start/end special tokens
                        target_vocab=data.tokenizer_en.vocab_size + 2, # same
                        max_seq_input=max_len,
                        max_seq_target=max_len,
                        drop_rate=0.1)

    learning_rate = CustomSchedule(dm, warmup_steps=4000)
    optimizer = K.optimizers.Adam(beta_1=0.9,
                                  beta_2=0.98,
                                  epsilon=1e-9)
    loss_object = K.losses.SparseCategoricalCrossentropy(from_logits=True) # NOTE Ignore mask ?

    # model.compile(optimizer=optimizer,
    #               loss=loss_object,
    #               run_eagerly=True,  # NOTE Remains to be seen
    #               metrics=['loss', 'accuracy'])

    # Define metrics
    train_loss = K.metrics.Mean(name='train_loss')
    train_accuracy = K.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    # Training step function
    @tf.function
    def train_step(inp, tar):

        enc_mask, combined_mask, dec_mask = create_masks(inp, tar)

        with tf.GradientTape() as tape:
            predictions, _ = model(inp, tar, True, enc_mask, combined_mask, dec_mask)
            loss = loss_object(tar, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(tar, predictions)

    # Training loop
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1} started...')
        train_loss.reset_state()
        train_accuracy.reset_state()

        for (batch, (inp, tar)) in enumerate(data.data_train):
            train_step(inp, tar)

            if batch % 50 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch}: Loss {train_loss.result()}, '
                      f'Accuracy {train_accuracy.result()}')

        print(f'Epoch {epoch + 1}: Loss {train_loss.result()}, '
              f'Accuracy {train_accuracy.result()}')

    return model
