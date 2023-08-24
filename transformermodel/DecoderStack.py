"""
Represents the Decoder part of the Transformer model, which is composed of a stack of identical layers.

Created Aug 24 16:44:16 2023

@author: lmienhardt, lkuehl
"""

# logging and decorators
import logging as log

#from tensorflow import keras
from tensorflow.python.keras import layers

# impport from other documents
from utilities.clones import clones
from LayerNorm import LayerNorm

class DecoderStack(layers.Layer):
    """
    This class represents the Decoder part of the Transformer model, which is composed of a stack of identical layers.
    Each layer in the Decoder Stack consists of three sub-layers: 
        - a Masked Multi-head Self-Attention mechanism, 
        - a Multi-head Self-Attention mechanism over the Encoder's output,
        - a Position-wise Fully Connected Feed-Forward network.
    A residual connection is employed around each of the three sub-layers, followed by layer normalization.
    """

    def __init__(self, layer, N, size, **kwargs):
        """
        Inititalize the DecoderStack instance
        Args:
            layer (layers.layer):   An instance of a layer, which will be cloned N times to form the decoder stack.
            N (int):                The number of layers in the decoder stack.
            size (int):             The dimensionality of the input/output space of the decoder.
            **kwargs (various):     Additional keyword arguments. They contain the parameters of the layers in self.layers, such that they can
                                    be passed to the clone function that initializes the layers.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()
        self.layers = clones(layer, N, size=size, **kwargs)
        self.norm = LayerNorm(size)

    def call(self, input_tensor, enc_memory_tensor, src_mask, tgt_mask, training=None):
        """
        This function propagates the input through the decoder stack, by applying succesively each layer in the self.layers attribute.
        This is equivalent to running the masked attention layer, the attention layer over the encoder's output, 
        and the fully connected feed-forward layer N times, before finally normalising and returning an output.

        Args:
            input_tensor (Tensor):                  The input to the decoder.
            enc_memory_tensor (Tensor):             The output of the encoder, serves as the "memory" in the Transformer model.
            src_mask (Tensor of dtype tf.Boolean):  A boolean mask tensor for padding positions within the source input.
            tgt_mask (Tensor of dtype tf.Boolean):  A boolean mask tensor for padding and preventing "future" information 
                                                    in attenting to the source input.
            training (bool, None):                  A boolean indicating whether to run the layer in training mode or inference mode.

        Returns:
            decoder_out (Tensor):                   The output of the decoder stack.
        """
        for layer in self.layers:
            input_tensor = layer(input_tensor, 
                                 enc_memory_tensor, 
                                 src_mask, tgt_mask, 
                                 training=training)
            
        decoder_out = self.norm(input_tensor, training=training)
        
        return decoder_out