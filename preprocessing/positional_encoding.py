"""
Each position in the input sequence is assigned a unique vector representation.

Created Aug 23 16:55:07 2023

@author: lmienhardt, lkuehl
"""

# logging and decorators
import logging as log

# general modules
import numpy as np

# tensorflow modules
import tensorflow as tf

def positional_encoding(length, depth):
    """
    Generate positional encoding for a given length and depth to provide positional information.
    Positional encoding is a technique where each position in the input sequence is assigned a 
    unique vector representation.

    The encoding vector alternates between the sine and cosine functions of different 
    frequencies, which allows the model to distinguish the position of the inputs.

    The positional encoding function uses a specific ratio to scale down the angle rates 
    exponentially (1 / (10000**(depth/depth))). It means that for lower dimensions in the 
    positional encoding, the angle rate is high which means the positional encoding is 
    changing rapidly for lower dimensions. For higher dimensions, the angle rate is low 
    which means the positional encoding is changing slowly. It gives a balance between 
    low and high frequency information.

    Args:
        length (int):   Length of the sequence for which positional encoding is to be generated.
        depth (int):    The number of dimensions for the positional encoding. Equals the embedding size.

    Returns:
        Tensor:         A 2D Tensor of shape (length, depth) containing the positional encoding vectors.
    """
    log.debug(f'execute')
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]    # Creates a numpy array of shape (sequence_length, 1)
                                                    # filled with the numbers 1 to sequence length
    depths = np.arange(depth)[np.newaxis, :]/depth  # Creates a numpy array of shape (1, depth/2)
                                                    # filled with the numbers 1 to depth/2 divided by depth

    angle_rates = 1 / (10000**depths) 
    angle_rads  = positions * angle_rates           # broadcasting such that now element [i,j] is pos(i) * angle(j)

    # as we have above chosen depth/2 we can now concatenate sines and cosines to aquire an vectore of size depth
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)