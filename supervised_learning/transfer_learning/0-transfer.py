#!/usr/bin/env python3

"""
Transfer Knowledge
"""

# from tensorflow import keras as K

import keras as K


def preprocess_data(X, Y):
    """
    """
    # NOTE using dedicated preprocessing function
    X_p = K.applications.densenet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, num_classes=10)
    return X_p, Y_p


if __name__ == "__main__":
    from keras.src.applications.densenet import DenseNet121
    # load data
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    # preprocessing for DenseNet
    X_train, y_train = preprocess_data(x_train, y_train)
    X_test, y_test = preprocess_data(x_test, y_test)

    # Resize the input
    inputs = K.Input(shape=(32, 32, 3))
    resized_input = K.layers.Lambda(lambda image: K.ops.image.resize(image, (224, 224)))(inputs)

    # create the base pre-trained model
    # NOTE might have to set input_shape
    base_model = DenseNet121(weights='imagenet',
                             include_top=False)
    base_model.trainable = False

    # add a global spatial average pooling layer
    x = base_model.output
    x = K.layers.GlobalAveragePooling2D()(x)

    # let's add a fully-connected layer
    x = K.layers.Dense(300, activation='relu',
                       kernel_initializer=K.initializers.HeNormal(seed=0),
                       kernel_regularizer=K.regularizers.L2())(x)
    x = K.layers.Dropout(0.3)(x)

    # logistic layer -- we have 10 classes
    predictions = K.layers.Dense(10, activation='softmax')(x)

    # this is the model we will train
    model = K.Model(inputs=base_model.input, outputs=predictions)

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # train the model on the new data for a few epochs
    model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test),
              batch_size=300, epochs=10, verbose=1)

    model.summary()
