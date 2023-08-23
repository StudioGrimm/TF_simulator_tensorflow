"""
Computes loss on a batch of examples.

Created Aug 23 17:47:29 2023

@author: lmienhardt, lkuehl
"""

# logging and decorators
import logging as log

# tensorflow modules
import tensorflow as tf

class LossCompute(tf.keras.losses.Loss):
    """
    Custom loss computation class that computes loss on a batch of examples.
    This class inherits from tf.keras.losses.Loss, which allows it to be used seamlessly 
    within the Keras API.
    """
    def __init__(self, generator, loss_function, vocab_size, name='loss_compute'):
        """
        Initializes the LossCompute object.
        
        Args:
            generator (layers.Layer):       The generator layer.
            loss_function (layers.Layer):   The function class to compute the loss.
            vocab_size (int):               The size of the vocabulary.
            name (str, optional):           The name for the loss.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__(name=name)
        self.generator = generator
        self.loss_function = loss_function
        self.vocab_size = vocab_size

    def call(self, y_true, y_pred):
        """
        Computes the loss on a batch of examples.
        
        Args:
            y_true (tf.Tensor): The true labels.
            y_pred (tf.Tensor): The predicted labels.

        Returns:
            tf.Tensor: The total loss for the batch.
        """
        # generate predictions as one-hot encoded tensor
        y_pred = self.generator(y_pred)
        y_true_one_hot = tf.cast(tf.one_hot(y_true, depth=self.vocab_size), tf.float32)

        # Compute loss
        loss = self.loss_function(y_pred, y_true_one_hot)

        # Calculate mean loss per batch
        norm = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
        sloss = loss / norm

        # Return total loss (for the whole batch)
        # TODO: Do we want mean loss or total loss?
        return sloss