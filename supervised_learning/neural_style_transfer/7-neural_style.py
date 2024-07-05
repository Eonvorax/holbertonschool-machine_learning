#!/usr/bin/env python3
"""
Neural Style Transfer
"""


import numpy as np
import tensorflow as tf


class NST:
    """
    The NST class performs tasks for neural style transfer.

    Public Class Attributes:
    - style_layers: A list of layers to be used for style extraction,
    defaulting to ['block1_conv1', 'block2_conv1', 'block3_conv1',
    'block4_conv1', 'block5_conv1'].
    - content_layer: The layer to be used for content extraction,
    defaulting to 'block5_conv2'.
    """
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializes an NST instance.

        Parameters:
        - style_image (numpy.ndarray): The image used as a style reference.
        - content_image (numpy.ndarray): The image used as a content reference
        - alpha (float): The weight for content cost. Default is 1e4.
        - beta (float): The weight for style cost. Default is 1.

        Raises:
        - TypeError: If style_image is not a numpy.ndarray with
            shape (h, w, 3), raises an error with the message "style_image
            must be a numpy.ndarray with shape (h, w, 3)".
        - TypeError: If content_image is not a numpy.ndarray with
            shape (h, w, 3), raises an error with the message "content_image
            must be a numpy.ndarray with shape (h, w, 3)".
        - TypeError: If alpha is not a non-negative number, raises an error
            with the message "alpha must be a non-negative number".
        - TypeError: If beta is not a non-negative number, raises an error
            with the message "beta must be a non-negative number".

        Instance Attributes:
        - style_image: The preprocessed style image.
        - content_image: The preprocessed content image.
        - alpha: The weight for content cost.
        - beta: The weight for style cost.
        """
        if (not isinstance(style_image, np.ndarray)
                or style_image.shape[-1] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray)
                or content_image.shape[-1] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(alpha, (float, int)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (float, int)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixel values are between 0 and 1
        and its largest side is 512 pixels.

        Parameters:
        - image (numpy.ndarray): A numpy.ndarray of shape (h, w, 3) containing
        the image to be scaled.

        Raises:
        - TypeError: If image is not a numpy.ndarray with shape (h, w, 3),
          raises an error with the message "image must be a numpy.ndarray
          with shape (h, w, 3)".

        Returns:
        - tf.Tensor: The scaled image as a tf.Tensor with shape
          (1, h_new, w_new, 3), where max(h_new, w_new) == 512 and
          min(h_new, w_new) is scaled proportionately.
          The image is resized using bicubic interpolation, and its pixel
          values are rescaled from the range [0, 255] to [0, 1].
        """
        if (not isinstance(image, np.ndarray) or image.shape[-1] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w = image.shape[:2]

        if w > h:
            new_w = 512
            new_h = int((h * 512) / w)
        else:
            new_h = 512
            new_w = int((w * 512) / h)

        # Resize image (with bicubic interpolation)
        image_resized = tf.image.resize(
            image, size=[new_h, new_w],
            method=tf.image.ResizeMethod.BICUBIC)

        # Normalize pixel values to the range [0, 1]
        image_normalized = image_resized / 255

        # Clip values to ensure they are within [0, 1] range
        image_clipped = tf.clip_by_value(image_normalized, 0, 1)

        # Add batch dimension on axis 0 and return
        return tf.expand_dims(image_clipped, axis=0)

    def load_model(self):
        """
        Load the VGG19 model with AveragePooling2D layers instead of
        MaxPooling2D layers.
        """
        # get VGG19 from Keras API
        modelVGG19 = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        modelVGG19.trainable = False

        # Selected layers
        selected_layers = self.style_layers + [self.content_layer]

        outputs = [modelVGG19.get_layer(name).output for name
                   in selected_layers]

        # Construct model
        model = tf.keras.Model([modelVGG19.input], outputs)

        # replace MaxPooling layers by AveragePooling layers
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        tf.keras.models.save_model(model, 'vgg_base.h5')
        model_avg = tf.keras.models.load_model('vgg_base.h5',
                                               custom_objects=custom_objects)

        self.model = model_avg

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculate the Gram matrix of a given tensor.

        Args:
        - input_layer: A tf.Tensor or tf.Variable of shape (1, h, w, c).

        Returns:
        - A tf.Tensor of shape (1, c, c) containing the Gram matrix of
            input_layer.
        """
        # Calidate input_layer rank and batch size
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable))
                or len(input_layer.shape) != 4
                or input_layer.shape[0] != 1):
            raise TypeError("input_layer must be a tensor of rank 4")

        # calculate gram matrix: (batch, height, width, channel)
        gram = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)

        # Normalize by number of locations (h * w) then return gram tensor
        input_shape = tf.shape(input_layer)
        nb_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return gram / nb_locations

    def generate_features(self):
        """
        Extract the features used to calculate neural style cost.
        Sets the public instance attributes:
            - gram_style_features - a list of gram matrices calculated from the
                style layer outputs of the style image
            - content_feature - the content layer output of the content image
        """
        # Ensure the images are preprocessed correctly
        preprocessed_style = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        preprocessed_content = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)

        # Get the outputs from the model with preprocessed images as input
        style_outputs = self.model(preprocessed_style)[:-1]

        # Set content_feature, no further processing required
        self.content_feature = self.model(preprocessed_content)[-1]

        # Compute and set Gram matrices for the style layers outputs
        self.gram_style_features = [self.gram_matrix(
            output) for output in style_outputs]

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer

        - style_output - tf.Tensor of shape (1, h, w, c) containing the layer
            style output of the generated image
        - gram_target - tf.Tensor of shape (1, c, c) the gram matrix of the
            target style output for that layer

        Returns: the layer's style cost
        """
        if (not isinstance(style_output, (tf.Tensor, tf.Variable))
                or len(style_output.shape) != 4):
            raise TypeError("style_output must be a tensor of rank 4")

        c = style_output.shape[-1]

        if (not isinstance(gram_target, (tf.Tensor, tf.Variable))
                or gram_target.shape != (1, c, c)):
            raise TypeError(
                f"gram_target must be a tensor of shape [1, {c}, {c}]")

        # Calculate the Gram matrix of style_output
        gram_style_output = self.gram_matrix(style_output)

        # Return squared mean of the difference between the two Gram matrices
        return tf.reduce_mean(tf.square(gram_style_output - gram_target))

    def style_cost(self, style_outputs):
        """
        Calculates the style cost for generated image

        - style_outputs - a list of tf.Tensor style outputs for the
            generated image

        Returns: the style cost
        """
        l_s = len(self.style_layers)

        if not isinstance(style_outputs, list) or len(style_outputs) != l_s:
            raise TypeError(
                f"style_outputs must be a list with a length of {l_s}")

        # Calc. the style cost for each layer with even weights
        style_cost = 0
        weight = 1.0 / float(l_s)
        for style_output, gram in zip(style_outputs, self.gram_style_features):
            style_cost += weight * self.layer_style_cost(style_output, gram)

        return style_cost

    def content_cost(self, content_output):
        """
        Calculates the content cost for the generated image

        - content_output - a tf.Tensor containing the content output for
            the generated image

        Returns: the content cost
        """
        s = self.content_feature.shape
        if (not isinstance(content_output, (tf.Tensor, tf.Variable))
                or s != content_output.shape):
            raise TypeError(f"content_output must be a tensor of shape {s}")

        return tf.reduce_mean(tf.square(content_output - self.content_feature))

    def total_cost(self, generated_image):
        """
        Calculates the total cost for the generated image

        - `generated_image` - a tf.Tensor of shape `(1, nh, nw, 3)` containing
            the generated image

        Returns: `(J, J_content, J_style)`
        - `J` is the total cost
        - `J_content` is the content cost
        - `J_style` is the style cost
        """
        s = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable))
                or s != generated_image.shape):
            raise TypeError(f"generated_image must be a tensor of shape {s}")

        # Preprocess the generated image
        preprocessed_img = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255.0)

        # Get model outputs of style and content layers for the generated image
        outputs = self.model(preprocessed_img)
        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        J_content = self.content_cost(content_output)
        J_style = self.style_cost(style_outputs)
        J = self.alpha * J_content + self.beta * J_style
        return J, J_content, J_style
