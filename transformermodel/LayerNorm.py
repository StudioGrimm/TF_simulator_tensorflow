"""
Implements the Layer Normalization technique.

Created Aug 24 13:40:48 2023

@author: lmienhardt, lkuehl
"""

# logging and decorators
import logging as log

# tensorflow modules
import tensorflow as tf
#from tensorflow import keras
from tensorflow.python.keras import layers

class LayerNorm(layers.Layer):
    """
    Implements the Layer Normalization technique, a type of normalization performed on inputs across features.

    Inherits from both the TensorFlow Keras Layer class for building custom layers, and a custom VisualWrapper class for data visualization.
    """

    def __init__(self, input_size, eps=1e-6):
        """
        Args:
        features (int):             The size of the input data.
            eps (float, optional):  A small constant added to the variance to avoid dividing by zero. Defaults to 1e-6.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()

        # Initialize the scale and offset parameters
        self.a_2 = self.add_weight(shape=(input_size,), initializer='ones', name=self.name + "a_2")
        self.b_2 = self.add_weight(shape=(input_size,), initializer='zeros', name=self.name + "b_2")
        self.eps = eps

    def call(self, input_tensor):
        """
        Performs the layer normalization on the input data.

        Args:
            input_tensor (Tensor):  The input data.

        Returns:
            norm_out (Tensor):      The normalized data.
        """
        # Compute the mean and variance of the input data
        mean, var = tf.nn.moments(input_tensor, axes=-1, keepdims=True)

        # Compute the standard deviation
        std = tf.math.sqrt(var + self.eps)

        # Perform the layer normalization
        norm_out = self.a_2 * (input_tensor - mean) / std + self.b_2

        return norm_out