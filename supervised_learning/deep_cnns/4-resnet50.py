#!/usr/bin/env python3
"""
ResNet-50
"""

from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in
    'Deep Residual Learning for Image Recognition (2015)'.

    All convolutions inside and outside the blocks are followed by
    batch normalization along the channels axis and a rectified linear
    activation (ReLU), respectively. All weights use he normal initialization
    with the seed set to zero.

    Returns:
        model (tf.keras.Model): The ResNet-50 model.
    """
    # Initializer he_normal with seed 0
    init = K.initializers.HeNormal(seed=0)
    # input tensor (assuming given shape)
    input_data = K.Input(shape=(224, 224, 3))

    # conv1 layer
    conv1 = K.layers.Conv2D(filters=64,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding="same",
                            kernel_initializer=init)(input_data)
    norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    activ1 = K.layers.Activation("relu")(norm1)

    # conv2_x layer
    maxpool_1 = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                   padding="same")(activ1)

    id_1_a = projection_block(maxpool_1, [64, 64, 256], s=1)
    proj_1_b = identity_block(id_1_a, [64, 64, 256])
    proj_1_c = identity_block(proj_1_b, [64, 64, 256])

    # conv3_x layer
    id_2_a = projection_block(proj_1_c, [128, 128, 512], s=2)
    proj_2_b = identity_block(id_2_a, [128, 128, 512])
    proj_2_c = identity_block(proj_2_b, [128, 128, 512])
    proj_2_d = identity_block(proj_2_c, [128, 128, 512])

    # conv4_x layer
    id_3_a = projection_block(proj_2_d, [256, 256, 1024], s=2)
    proj_3_b = identity_block(id_3_a, [256, 256, 1024])
    proj_3_c = identity_block(proj_3_b, [256, 256, 1024])
    proj_3_d = identity_block(proj_3_c, [256, 256, 1024])
    proj_3_e = identity_block(proj_3_d, [256, 256, 1024])
    proj_3_f = identity_block(proj_3_e, [256, 256, 1024])

    # conv5_x layer
    id_4_a = projection_block(proj_3_f, [512, 512, 2048], s=2)
    proj_4_b = identity_block(id_4_a, [512, 512, 2048])
    proj_4_c = identity_block(proj_4_b, [512, 512, 2048])

    # Average Pooling
    avg_pool = K.layers.AvgPool2D(pool_size=(7, 7), strides=(1, 1))(proj_4_c)

    # Fully Connected Layer, softmax
    dense_softmax = K.layers.Dense(units=1000, activation='softmax',
                                   kernel_initializer=init)(avg_pool)

    return K.Model(inputs=input_data, outputs=dense_softmax)
