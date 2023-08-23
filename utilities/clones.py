"""
Produces N identical layers.

Created Aug 23 16:33:15 2023

@author: lmienhardt, lkuehl
"""

# logging and decorators
import logging as log

def clones(layer_class, N, **kwargs):
    """Produce N identical layers"""
    log.debug(f'execute with class {layer_class.__class__.__name__} and N={N}')
    return [layer_class(**kwargs) for layer_number in range(N)]