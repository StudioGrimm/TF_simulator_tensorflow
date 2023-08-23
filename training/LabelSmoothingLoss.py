"""
Loss function layer that applies label smoothing.

Created Aug 23 17:43:22 2023

@author: lmienhardt, lkuehl
"""

# logging and decorators
import logging as log

# tensorflow modules
import tensorflow as tf

#from tensorflow import keras
from tensorflow.python.keras import layers

class LabelSmoothingLoss(layers.Layer):
    """
    This class represents a loss function layer that applies label smoothing to prevent overconfidence 
    in the model's predictions. This is done by replacing the 0s and 1s in the labels with smoothed values, 
    such that the model learns to be less confident and thus, more robust.

    Methods:
        call(x, target): Calculates and returns the loss given the model's output `x` and the target labels.

    Example:
        >>> loss_func = LabelSmoothingLoss(vocab_size=5000, padding_idx=0, smoothing=0.1)
        >>> x = tf.random.uniform((10, 5000))  # model's output
        >>> target = tf.random.uniform((10, 1), maxval=5000, dtype=tf.int32)  # target labels
        >>> loss = loss_func(x, target)  # calculate loss
    """

    def __init__(self, vocab_size, padding_idx, smoothing=0.0):
        """
        Args:
            vocab_size (int): The size of the vocabulary, which also represents the number of classes.
            padding_idx (int): The index representing padding elements.
            smoothing (float): The smoothing factor to be applied. The values should be between 0 and 1. 
                            Default value is 0.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx

        # confidence is used for the position of the predicted token, while smoothing is applied to all not predicted tokens
        self.confidence = 1.0 - smoothing   # value for prediction
        self.smoothing = smoothing          # value for smoothing
        
    def call(self, prediction, target):
        """
        This function applies label smoothing to the target labels, computes the KL divergence loss 
        between the predicted and smoothed target distributions, then masks out the padding tokens 
        in the loss (since those should not contribute to the training signal). Finally, it averages 
        the loss over the non-padding tokens.

        Args:
            prediction (tf.Tensor):     The predicted token logits from the model in form of a one-hot-encoding tensor.
                                        Shape is [batch_size, sequence_length, vocab_size].
            target (tf.Tensor):         The target token IDs. Shape is [batch_size, sequence_length].

        Returns:
            loss (tf.Tensor):           The average loss (scalar) for the given batch.

        Note:
            The loss is averaged over non-padding tokens.
        """
        # create padding mask
        mask = self.padding_mask(target, self.padding_idx)

        # Apply label confidence
        true_dist = target * self.confidence

        # Apply label smoothing
        smoothing_value = self.smoothing / tf.cast(self.vocab_size - 2, tf.float32)
        true_dist = tf.where(tf.equal(true_dist, 0), smoothing_value, true_dist)

        # Calculate the loss
        kl_div_loss = self.kl_div_loss(prediction, true_dist)
        masked_loss = tf.cast(self.apply_mask(kl_div_loss, mask), prediction.dtype)
        loss = tf.reduce_sum(masked_loss)/tf.reduce_sum(mask)

        return loss
    
    @staticmethod
    def padding_mask(tensor, padding_idx):
        """
        Create a binary mask where padding entries are 0 and others are 1.

        Args:
            tensor (tf.Tensor):     A tensor to be masked, of any shape.
            padding_idx (int):      The value that represents padding in the tensor.

        Returns:
            tf.Tensor:              A binary mask of the same shape as input tensor.
        """
        return tf.cast(tf.equal(tensor[:, :, padding_idx], 0), tf.float32)

    @staticmethod
    def apply_mask(tensor, mask):
        """
        Applies a mask to a tensor, zeroing out where the mask is on.

        Args:
            tensor (tf.Tensor):     A tensor to be masked, of any shape.
            mask (tf.Tensor):       A mask tensor, must be broadcastable to the shape of 'tensor'.

        Returns:
            tf.Tensor:              A tensor of the same shape as input tensor but with masked values zeroed out.
        """
        # mask stores padding bools in [batch_size, sequence_length] shape, 
        # we need to extend it to have shape [batch_size, sequence_length, vocab_size]
        expanded_mask = tf.broadcast_to(tf.expand_dims(mask, -1), tf.shape(tensor))

        return tensor * expanded_mask
    
    @staticmethod
    def kl_div_loss(input, target):
        """
        Calculates the Kullback-Leibler divergence between the input and target distributions.

        Notes: Inputs have to be logits, while target have to be probabilities.

        Args:
            input (tf.Tensor):      Input tensor, representing predicted probability distribution.
            target (tf.Tensor):     Target tensor, representing true probability distribution.

        Returns:
            tf.Tensor:              The KL divergence between the input and target distributions.
        """
        return target * (tf.math.log(target)-input)