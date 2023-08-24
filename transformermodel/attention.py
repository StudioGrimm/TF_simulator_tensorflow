"""
Computes 'Scaled Dot Product Attention'.

Created Aug 24 16:56:59 2023

@author: lmienhardt, lkuehl
"""

# logging
import logging as log

# tensorflow modules
import tensorflow as tf

def attention(query, key, value, mask=None, dropout=None, training=None):
    """
    Compute 'Scaled Dot Product Attention'

    The attention function computes a weighted sum of the value vectors, with the weights being determined by the similarity of the
    query vector with the corresponding key vector. The dot product of the query and key serves as the measure of similarity, and is scaled
    by the square root of the dimension of the key vector to avoid the issue of the dot product growing large for large dimensions.

    Args:
        query, key, value (Tensor):                     The query, key and value vectors. 
                                                        These typically have shape (batch_size, num_heads, seq_len, depth).
                                                        (seq_len as we want to calculate the attention for each position simultaneously)
        mask (Tensor of dtype tf.Boolean, optional):    A mask to apply to the attention scores before softmax, 
                                                        in order to prevent attention to certain positions. 
                                                        The shape should be broadcastable to shape (batch_size, num_heads, seq_len, seq_len???).
        dropout (Dropout, optional):                    Dropout layer to apply to the attention scores after softmax.
        training (bool, optional):                      Whether the model is in training mode.

    Returns:
        output (Tensor):                                The result of applying attention mechanism to the value vectors.
        p_attn (Tensor):                                The attention weights after applying softmax and dropout.
    """
    log.debug(f'execute')
    # Compute the dot product of the query and key vectors and scale by sqrt(d_k)
    d_k = tf.cast(query.shape[-1], dtype=tf.float32)
    scores = tf.matmul(query, tf.transpose(key, perm=[0, 1, 3, 2])) / tf.sqrt(d_k)

    # Apply the mask to the scores before softmax
    if mask is not None:
        mask = tf.cast(mask, dtype=tf.bool)
        scores = tf.where(mask, scores, tf.fill(tf.shape(scores), -1e9))

    # Apply softmax to the scores to get the attention weights
    p_attn = tf.nn.softmax(scores, axis=-1)

    # Apply dropout to the attention weights
    if dropout is not None:
        p_attn = dropout(p_attn, training=training)

    # Compute the weighted sum of the value vectors, using the attention weights
    attn_out = tf.matmul(p_attn, value)

    return attn_out, p_attn