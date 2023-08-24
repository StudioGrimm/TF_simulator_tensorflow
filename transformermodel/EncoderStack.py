"""
 A layer that applies a sublayer to the input, followed by dropout, and then adds the input to the result.

Created Aug 24 13:48:56 2023

@author: lmienhardt, lkuehl
"""

# logging and decorators
import logging as log

# from tensorflow import keras
from tensorflow.python.keras import layers

# import from other documents
from utilities.clones import clones
from LayerNorm import LayerNorm

class EncoderStack(layers.Layer):
    """
    This class represents the Encoder part of the Transformer model, which is composed of a stack of identical layers.
    Each layer in the Encoder Stack consists of two sub-layers: a Multi-head Self-Attention mechanism, and a Position-wise
    Fully Connected Feed-Forward network.
    A residual connection is employed around each of the two sub-layers, followed by layer normalization.
    """

    def __init__(self, layer, N, size, **kwargs):
        """
        Inititalize the EncoderStack instance
        Args:
            layer (layers.layer):   An instance of a layer, which will be cloned N times to form the encoder stack.
            N (int):                The number of layers in the encoder stack.
            size (int):             The dimensionality of the input/ouput space of the encoder.
            **kwargs (various):     Additional keyword arguments. They contain the parameters of the layers in self.layers, such that they can
                                    be passed to the clone function that initializes the layers.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()
        self.layers = clones(layer, N, size=size, **kwargs) # Creating N identical layer stacks to form the encoder
        self.norm = LayerNorm(size)

    def call(self, input_tensor, mask, training=None):
        """
        This function propagates the input through the encoder stack, by applying succesively each layer in the self.layers attribute.
        This is aquivalent to running the attention layer and the fully connected feed-forward layer N times, 
        before finally normalising and returning an output.

        Args:
            input_tensor (Tensor): The input to the encoder.
            mask (Tensor of dtype tf.Boolean): A boolean mask tensor for padding positions within the input.
            training (bool, None): A boolean indicating whether to run the layer in training mode or inference mode.

        Returns:
            (Tensor):              The output of the encoder stack.
        """
        for layer in self.layers:
            input_tensor = layer(input_tensor, mask, training=training)

        encoder_out = self.norm(input_tensor, training=training)

        return encoder_out