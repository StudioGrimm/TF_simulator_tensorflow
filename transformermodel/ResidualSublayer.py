"""
 A layer that applies a sublayer to the input, followed by dropout, and then adds the input to the result.

Created Aug 24 13:44:06 2023

@author: lmienhardt, lkuehl
"""

# logging and decorators
import logging as log

# from tensorflow import keras
from tensorflow.python.keras import layers

# import from other documents
from LayerNorm import LayerNorm

class ResidualSublayer(layers.Layer):
    """
    A layer that applies a sublayer to the input, followed by dropout, and then adds the input to the result.
    This follows the 'pre-norm' variation of the Transformer architecture, where Layer Normalization is applied before the sublayer.

    !!! This layer is used to wrap the attention sublayer and the feedforward layer in the encoder stack and decoder stack. !!!
    
    Inherits from both the TensorFlow Keras Layer class for building custom layers, and a custom VisualWrapper class for data visualization.
    """

    def __init__(self, size, dropout):
        """
        Args:
            size (int):         The number of features in the input data.
            dropout (float):    The rate of dropout to apply after the sublayer.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()

        self.norm = LayerNorm(size)
        self.dropout = layers.Dropout(dropout)

    def call(self, input_tensor, sublayer, training=None):
        """
        Applies the sublayer to the input after normalization, applies dropout, and then adds the input to the result.

        Args:
            input_tensor (Tensor):      The input data.
            sublayer (layers.Layer):    The sublayer to apply to the input data.
            training (bool):            Indicates whether the layer should behave in training mode (apply dropout) or in inference mode (do not apply dropout).

        Returns:
            residual_out (Tensor): The output data.
        """
        # Apply normalisation and sublayer
        norm_input = self.norm(input_tensor)
        sublayer_out = sublayer(norm_input)

        sublayer_dropout = self.dropout(sublayer_out, training=True)
            
        # compute residual output by applying dropout to the sublayer output and adding to the input
        residual_out = input_tensor + self.dropout(sublayer_out, training=training)

        return residual_out