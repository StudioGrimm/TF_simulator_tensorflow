"""
Computes 'Scaled Dot Product Attention'.

Created Aug 24 17:00:43 2023

@author: lmienhardt, lkuehl
"""

# logging 
import logging as log

# tensorflow modules
import tensorflow as tf
#from tensorflow import keras
from tensorflow.python.keras import layers

# import from other documents
from utilities.clones import clones
from attention import attention

class MultiHeadedAttention(layers.Layer):
    """
    MultiHeadedAttention layer is a key part of the Transformer model, enabling it to pay attention to different parts of the input for each output.
    This is achieved by having multiple 'heads', each of which independently computes a scaled dot product attention over the input.
    The outputs of these heads are then concatenated and linearly transformed to produce the final output.

    Attributes:
        d_k (int):                          Dimensionality of the query, key, and value vectors, 
                                            which should be identical for each head.
        h (int):                            Number of heads.
        query, key, value, linear (Dense):  These are the layers that perform the linear transformations for the input.
        attn (Tensor, optional):            Tensor storing the attention values from the last forward pass.
        dropout (Dropout):                  Dropout layer applied after the attention.
    """
    
    def __init__(self, h, d_model, dropout=0.1):
        """
        Args:
            h (int):                    Number of attention heads.
            d_model (int):              Dimensionality of the model.
            dropout (float, optional):  Dropout rate.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()

        assert d_model % h == 0 # make sure the number of attention heads are such that they can equally distribute over the input tensor
        self.d_k = d_model // h # determine the size for the attention heads
        self.h = h
        self.query, self.key, self.value, self.linear = clones(layers.Dense, N=4, units=d_model)
        self.attn = None
        self.dropout = layers.Dropout(dropout)

    def call(self, query, key, value, mask=None, training=None):
        """
        Forward pass for the MultiHeadedAttention layer.
        Applies linear transformations to the input, applies scaled dot product attention, then applies dropout, concatenates the heads,
        and applies a final linear transformation.

        Args:
            query, key, value (Tensor):                     Input tensors. Value and query are (in our case) always the same.
            mask (Tensor of dtype tf.Boolean, optional):    Boolean mask tensor for padding positions within the input.
            training (bool, optional):                      Indicates whether to run the layer in training mode or inference mode.

        Returns:
            result (Tensor):                                The output tensor.
        """

        if mask is not None:
            # Same mask applied to all h heads
            mask = tf.expand_dims(mask, 1)

        # find out how many batches are processed in parallel
        nbatches = tf.shape(query)[0]

        # Transform the input data into a shape that can be used as matrix input for the attention function.
        # The original size is d_model, the trainable matrices self.query, self.key, self.value transform this input tensor
        # into a tensor of the same size, but now we have to think of it as being of size h * d_k. Such that each section of size d_k,
        # will be passed through the attention mechanism independently. That is why all this transformations have to be done afterwards.
        # [nbatches, -1, self.h, self.d_k] does split the tensor into h smaller tensors of size d_k 
        # (nbatches is only there for working with batches of data). The Permutation ensures, that the h tensors are of shape (1, d_k), 
        # such that they can be processed.
        query, key, value = [
            tf.transpose(tf.reshape(lin_layer(input), [nbatches, -1 , self.h, self.d_k]), perm=[0, 2, 1, 3]) 
            for lin_layer, input in zip([self.query, self.key, self.value], (query, key, value))
        ]

        att_out, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout, training=training)

        # Now we reverse the whole process and reshape the output into vectors of shape (nbatches, 1, d_model) again.
        att_out = tf.reshape(tf.transpose(att_out, perm=[0, 2, 1, 3]), (nbatches, -1, self.h * self.d_k))

        # This finally mixes the results of the different heads together into one output vector
        linear_output = self.linear(att_out)

        return linear_output