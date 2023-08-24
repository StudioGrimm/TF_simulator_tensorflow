"""
Implements the Position-wise Feed-Forward Network (FFN) for the Transformer model.

Created Aug 24 16:50:50 2023

@author: lmienhardt, lkuehl
"""

# logging and decorators
import logging as log

# tensorflow modules
import tensorflow as tf
#from tensorflow import keras
from tensorflow.python.keras import layers

class PositionwiseFeedForward(layers.Layer):
    """
    Implements the Position-wise Feed-Forward Network (FFN) for the Transformer model.
    The FFN consists of two fully connected layers with a ReLU activation in between.

    Attributes:
        dense_in (Dense):    First dense layer.
        dense_out (Dense):   Second dense layer.
        dropout (Dropout):   Dropout layer.
    """


    def __init__(self, d_model, d_ff, dropout=0.1):
        """
            Args:
                d_model (int):      Output dimensionality of the Transformer.
                d_ff (int):         Inner-layer dimensionality.
                dropout (float):    Dropout rate after the ReLU activation.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()
        self.dense_in = layers.Dense(d_ff)
        self.dense_out = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout)

    def call(self, input_tensor, training=None):
        """
        Forward pass for the FFN. Applies both layers and ReLU activation to the input tensor.
        Dropout inbetween the layers, as the ResidualLayer that wraps around will again perform a dropout

        Args:
            x (Tensor): Input tensor.
            training (bool, optional): Indicates whether to run the layer in training mode or inference mode.

        Returns:
            (Tensor): Output tensor.
        """
        return self.dense_out(self.dropout(tf.nn.relu(self.dense_in(input_tensor)), training=training))