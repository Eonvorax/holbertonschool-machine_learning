#!/usr/bin/env python3
"""
Transfer Knowledge
"""

from tensorflow import keras as K


def preprocess_data(X, Y):
    """
    Pre-processes the data for your model:

    X is a numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data,
        where m is the number of data points
    Y is a numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X

    Returns: X_p, Y_p
        X_p is a numpy.ndarray containing the preprocessed X
        Y_p is a numpy.ndarray containing the preprocessed Y
    """
    # NOTE using dedicated preprocessing function
    X_p = K.applications.densenet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, num_classes=10)
    return X_p, Y_p


def build_model():
    """
    Builds a DenseNet model with 10 classes, using ImageNet weights.
    The top layers are all frozen except the last few layers, to allow
    for fine-tuning on CIFAR-10.
    The first layer added is a lambda layer that scales up the data to
    the correct size for CIFAR-10.
    The following layers are added for feature classification.
    """
    # from keras.api.applications.densenet import DenseNet121
    # NOTE might have to set input_shape
    base_model = K.applications.DenseNet121(weights='imagenet',
                                            include_top=False,
                                            classes=10,
                                            input_shape=(224, 224, 3))

    base_model.trainable = False
    # # Unfreeze the last block (last 10 layers)
    # for layer in base_model.layers[:-10]:
    #     layer.trainable = False

    # Functional API approach
    inputs = K.Input(shape=(32, 32, 3))
    resized_inputs = K.layers.Lambda(
        lambda x: K.preprocessing.image.smart_resize(x, (224, 224)))(inputs)

    # NOTE running in inference mode, needs testing
    base_model_output = base_model(resized_inputs, training=False)

    # x = K.layers.Flatten()(base_model_output)
    x = K.layers.GlobalAveragePooling2D()(base_model_output)
    x = K.layers.Dropout(0.3)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    return K.Model(inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    # Load data from CIFAR-10
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    # preprocessing for DenseNet
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    # Build and compile the new model, show summary
    model = build_model()
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # Callbacks: early stopping and model checkpoint
    early_stopping = K.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, restore_best_weights=False)
    checkpoint = K.callbacks.ModelCheckpoint(filepath='cifar10.h5',
                                             monitor='val_accuracy',
                                             save_best_only=True,
                                             mode='max')

    # Train the model
    history = model.fit(x_train, y_train,
                        epochs=10,
                        validation_data=(x_test, y_test),
                        callbacks=[early_stopping, checkpoint],
                        verbose=1)

    model.trainable = True
    model.compile(optimizer=K.optimizers.Adam(1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    finetuning_history = model.fit(x_train, y_train,
                                   epochs=10,
                                   validation_data=(x_test, y_test),
                                   callbacks=[early_stopping, checkpoint],
                                   verbose=1)

    # Save model in h5 format
    model.save(filepath='cifar10.h5', save_format='h5')

    # Evaluate on test set
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f'Test accuracy: {accuracy * 100:.2f}%')
