"""
Applies positional encoding on top of embeddings.

Created Aug 23 17:02:01 2023

@author: lmienhardt, lkuehl
"""

import logging as log

# tensorflow modules
import tensorflow as tf
#from tensorflow import keras
from tensorflow.python.keras import layers

# import definitions from other files
from positional_encoding import positional_encoding

class PositionalEmbedding(layers.Layer):
    """
    A Keras layer to apply positional encoding on top of embeddings.

    This layer creates embeddings for discret input vectors created by a tokenizer
    and applies positional encoding to these embeddings to provide positional information.
    The positional encoding is pre-computed in the constructor for efficiency and it is added to the output 
    of the embedding layer in the `call` method. The dropout is used to train the embeddings.
    """
    def __init__(self, vocab_size, d_model, dropout):
        """
        Initializes Positional Embeddings

        Args:
            vocab_size (int):   The size of the input token vector.
            d_model (int):      The dimension used for the embeddings and positional encoding passed to the model.
            dropout (float):    Value used for dropout.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.dropout = layers.Dropout(dropout)

        # calculate positional encoding
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def call(self, input_token_vec, training=None):
        """
        Performs the forward pass for the embedding and positional encoding layers.
        
        Args:
            input_token_vec (Tensor):   Input tensor of shape `(batch_size, sequence_length)`.
            training (bool, optional):  Indicator for the mode (training or inference) of the model.

        Returns:
            y (Tensor):     The output tensor after applying embedding, positional encoding, and dropout. 
                            It has the shape of `(batch_size, sequence_length, d_model)`.
        """

        length = tf.shape(input_token_vec)[1]

        x_emb = self.embedding(input_token_vec) # is now a tensor of shape (batch_size, length, d_model)
        x_emb_scale = x_emb * tf.math.sqrt(tf.cast(self.d_model, tf.float32)) # This factor sets the relative scale of the embedding and positional_encoding
        
        y = self.dropout(x_emb_scale + self.pos_encoding[tf.newaxis, :length, :])

        return y