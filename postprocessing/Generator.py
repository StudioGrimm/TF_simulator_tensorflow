"""
Generates the predicted output.

Created Aug 23 17:29:07 2023

@author: lmienhardt, lkuehl
"""

# logging and decorators
import logging as log

# tensorflow modules
import tensorflow as tf

#from tensorflow import keras
from tensorflow.python.keras import layers

class Generator(layers.Layer):
    """
    This class serves as the final layer of the Transformer model, generating the predicted output.
    It applies a dense layer to the final output of the Transformer model and then a log softmax function 
    across the vocabulary dimension. This results in a distribution over the possible output tokens for each 
    position in the sequence, where the value of each token is the log probability of that token being the 
    output for that position.

    Attributes:
        proj (Dense): Dense layer that is applied to the final output of the Transformer model. It increases 
        the dimensionality of the input to be the size of the vocabulary.
    """

    def __init__(self, vocab):
        """
        Args:
            vocab (int): Size of the output vocabulary.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()

        self.proj = layers.Dense(vocab)

    def call(self, input_tensor, training=None):
        """
        This method applies the Dense layer and log softmax function to its input.
        
        Args:
            input_tensor (Tensor):      The input tensor, which is the final output from the Transformer model.
            training (bool, optional):  Indicates whether to run the layer in training mode or inference mode.

        Returns:
            result (Tensor):    A tensor of the same shape as the input, but the last dimension is now the size 
                                of the vocabulary. Each value in this tensor is the log probability of the corresponding token 
                                being the output for the position in the sequence.
        """
        result = tf.nn.log_softmax(self.proj(input_tensor), axis=-1)

        return result