"""
Mask out subsequent positions.

Created Aug 23 16:38:18 2023

@author: lmienhardt, lkuehl
"""

import numpy as np

def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return subsequent_mask == 0