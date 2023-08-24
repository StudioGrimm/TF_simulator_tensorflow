"""
Represents a single layer within the Encoder stack of the Transformer model.

Created Aug 24 16:47:31 2023

@author: lmienhardt, lkuehl
"""

# logging and decorators
import logging as log

#from tensorflow import keras
from tensorflow.python.keras import layers

# import from other documents
from utilities.clones import clones
from ResidualSublayer import ResidualSublayer

class DecoderLayer(layers.Layer):
    """
    This class represents a single layer within the Decoder stack of the Transformer model.
    Each DecoderLayer consists of three sub-layers:
        - a Masked Multi-head Self-Attention mechanism,
        - a Multi-head Self-Attention mechanism that interacts with the output of the encoder,
        - a Position-wise Fully Connected Feed-Forward network.
    A residual connection is employed around each of the three sub-layers.
    
    Note: The residual sublayers do not themselves contain sublayers, because of two reasons:
      - This way, we can clone the ResidualSublayer, instead of having to write out each single sublayer.
      - During the forward pass, we need to pass different information to the sublayers e.g. masks, training, context. 
        This process is simplified if the ResidualSublayer can be skipped.
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        Initializes the DecoderLayer
        Args:
            size (int):                    The dimensionality of the input/output space of the decoder.
            self_attn (layers.Layer):      The Masked Multi-head Self-Attention mechanism.
            src_attn (layers.Layer):       The Masked Multi-head Source-Attention mechanism that interacts with the encoder output.
            feed_forward (layers.Layer):   The Position-wise Fully Connected Feed-Forward network.
            dropout (float):               The dropout rate to be applied to the output during training.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSublayer, N=3, size=size, dropout=dropout)

    def call(self, input_tensor, enc_memory, src_mask, tgt_mask, training=None):
        """
        This function propagates the input through the decoder layer.

        The Masked Self-Attention mechanism (sublayer[0]) takes three times the input_tensor as input for query, key, value.
        It hides the padding and future positions from affecting the current token's attention calculation through a combined padding and lookahead mask, 
        passed through the tgt_mask argument, and returns the rearranged output.

        The Encoder-Decoder Attention mechanism (sublayer[1]) takes the output from the previous Masked Self-Attention mechanism as the query and the encoder 
        output (memory) as both the key and the value. It also employs a padding mask (src_mask) on the encoder output and returns the attention-combined output.

        The Feed-Forward network (sublayer[2]) takes the output of the Encoder-Decoder Attention mechanism and creates meaningful information for the next Decoder-Layer.
        
        Args:
            input_tensor (Tensor):                             The input to the decoder layer.
            enc_memory (Tensor):                        The output of the encoder, serves as the "memory" in the Transformer model.
            src_mask (Tensor of dtype tf.Boolean):  A boolean mask tensor for padding positions within the source input.
            tgt_mask (Tensor of dtype tf.Boolean):  A boolean mask tensor for padding and preventing "future" information in self-attention mechanism within the target input.
            training (bool, optional):              A boolean indicating whether to run the layer in training mode or inference mode.

        Returns:
            ff_out (Tensor):                The output of the decoder layer.
        """
        self_attn_out = self.sublayer[0](input_tensor, 
                                         lambda x: self.self_attn(x, x, x, 
                                                                  tgt_mask, 
                                                                  training=training),
                                         training=training)
        src_attn_out = self.sublayer[1](self_attn_out, 
                                        lambda x: self.src_attn(x, enc_memory, enc_memory, 
                                                                src_mask,
                                                                training=training),
                                        training=training)
        ff_out = self.sublayer[2](src_attn_out, 
                                  lambda x: self.feed_forward(x,
                                                              training=training),
                                  training=training)

        return ff_out