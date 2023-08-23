"""
Custom learning rate scheduler for the Transformer model.

Created Aug 23 17:49:13 2023

@author: lmienhardt, lkuehl
"""

# logging and decorators
import logging as log

# tensorflow modules
import tensorflow as tf

class TransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom learning rate scheduler for the Transformer model.

    This class inherits from tf.keras.optimizers.schedules.LearningRateSchedule, which allows it to be used seamlessly
    within the Keras API for dynamically adjusting the learning rate during training.

    It follows the learning rate schedule defined in the "Attention is All You Need" paper, which increases the 
    learning rate linearly for the first 'warmup_steps', and decreases it afterwards proportionally to the inverse 
    square root of the step number.
    """
    def __init__(self, d_model=512, warmup_steps=4000):
        """
        Initializes the TransformerSchedule object.
        
        Args:
            d_model (int, optional):        The dimensionality of the input.
            warmup_steps (int, optional):   The number of steps for the linear warmup phase.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32) # for calculations we need float tensors
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """
        Computes the learning rate for a given step.
        
        Args:
            step (int): The current training step.

        Returns:
            tf.Tensor: The learning rate for the provided step.
        """
        step = tf.cast(step, dtype=tf.float32)  # convert for calculations
        arg_1 = tf.math.rsqrt(step)             # rsqrt is equivalent to 1/sqrt(x)
        arg_2 = step * (self.warmup_steps ** -1.5)
        
        # Minimum of two arguments provides the linear warmup phase for the first 'warmup_steps'
        # and the decrease proportional to the inverse square root of the step afterwards.
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg_1, arg_2)