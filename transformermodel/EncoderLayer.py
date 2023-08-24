"""
Represents a single layer within the Encoder stack of the Transformer model.

Created Aug 24 16:39:07 2023

@author: lmienhardt, lkuehl
"""

# logging and decorators
import logging as log

# from tensorflow import keras
from tensorflow.python.keras import layers

# import from other documents
from utilities.clones import clones
from ResidualSublayer import ResidualSublayer

class EncoderLayer(layers.Layer):
    """
    This class represents a single layer within the Encoder stack of the Transformer model.
    Each EncoderLayer consists of two sub-layers: 
        - a Multi-head Self-Attention mechanism, 
        - a Position-wise Fully Connected Feed-Forward network.
    A residual connection is employed around each of the two sub-layers.
    
    Note:   The residual sublayers do themselves not contain sublayers, because of two reasons:
                - Like that we can clone the ResidualSublayer, instead of having to write out each single sublayer
                - During forward pass, we need to pass different information to the sublayers e.g. mask, training, x, context.
                  This process is simplified if the ResidualSublayer can be skipped.
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        Initializes the EncoderLayer
        Args:
            size (int):                    The dimensionality of the input/output space of the encoder.
            self_attn (layers.Layer):      The Multi-head Self-Attention mechanism.
            feed_forward (layers.Layer):   The Position-wise Fully Connected Feed-Forward network.
            dropout (float):               The dropout rate to be applied to the output during training.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSublayer, 
                               N=2, 
                               size=size, 
                               dropout=dropout)

    def call(self, input_tensor, mask, training=None):
        """
        This function propagates the input through the attention layer and the feed-forward layer.

        The Self-Attention mechanism (sublayer[0]) takes three times the input_tensor as input for query, key, value
        it hiddes the padding through a padding mask, passed through the mask argument, and returns the rearranged output.

        The Feed-Forward network takes the output of the Self-Attention mechanism and creates meaningful information for the next Encoder-Layer.
        
        Args:
            input_tensor (Tensor):              The input to the encoder layer.
            mask (Tensor of dtype tf.Boolean):  A boolean mask tensor for padding positions within the input.
            training (bool, optional):          A boolean indicating whether to run the layer in training mode or inference mode.

        Returns:
            ff_output (Tensor):                 The output of the encoder layer.
        """
        attn_output = self.sublayer[0](input_tensor, 
                                        lambda x: self.self_attn(x, x, x, 
                                                                 mask, 
                                                                 training=training), 
                                        training=training)
        ff_output = self.sublayer[1](attn_output, 
                                     lambda x: self.feed_forward(x, 
                                                                 training=training), 
                                     training=training)
        return ff_output