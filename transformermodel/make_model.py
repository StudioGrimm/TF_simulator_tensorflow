"""
Creates the model.

Created Aug 24 17:20:11 2023

@author: lmienhardt, lkuehl
"""

# logging and decorators
import logging as log

# tensorflow modules
import tensorflow as tf

# import from other documents
from EncoderDecoder import EncoderDecoder
from EncoderStack import EncoderStack
from EncoderLayer import EncoderLayer
from DecoderStack import DecoderStack
from DecoderLayer import DecoderLayer
from MultiHeadedAttention import MultiHeadedAttention
from PositionwiseFeedForward import PositionwiseFeedForward
from preprocessing.PositionalEmbedding import PositionalEmbedding
from postprocessing.Generator import Generator

def make_model(src_vocab, 
               tgt_vocab, 
               N=6, 
               d_model=512, 
               d_ff=2048, 
               h=8, 
               dropout=0.1) -> tf.keras.Model:
    """
    Constructs a Transformer model from the given hyperparameters.

    Args:
        src_vocab (int):            The size of the source vocabulary.
        tgt_vocab (int):            The size of the target vocabulary.
        N (int, optional):          The number of layers in the Transformer's encoder and decoder stacks. Default is 6.
        d_model (int, optional):    The dimension of the Transformer's embedding space. Default is 512.
        d_ff (int, optional):       The dimension of the feed forward network model. Default is 2048.
        h (int, optional):          The number of attention heads. Default is 8.
        dropout (float, optional):  The dropout rate. Default is 0.1.

    Returns:
        model (tf.keras.Model): A Transformer model constructed from the provided hyperparameters.

    This function constructs an Encoder-Decoder model using the specified hyperparameters. 
    The Encoder and Decoder stacks each consist of N layers. 
    Each layer in the Encoder stack consists of a multi-headed self-attention mechanism, 
    followed by position-wise fully connected feed-forward network. 
    Each layer in the Decoder stack consists of a multi-headed self-attention mechanism, 
    a multi-headed source-attention mechanism over the Encoder's output, 
    and position-wise fully connected feed-forward network.
    """
    log.debug(f'execute')
    model = EncoderDecoder(
                EncoderStack(
                    EncoderLayer,
                    N=N, 
                    size=d_model, 
                    dropout=dropout, 
                    self_attn=MultiHeadedAttention(h, d_model), 
                    feed_forward=PositionwiseFeedForward(d_model, d_ff, dropout)),
                DecoderStack(
                    DecoderLayer, 
                    N=N, 
                    size=d_model, 
                    dropout=dropout,
                    self_attn=MultiHeadedAttention(h, d_model), 
                    src_attn=MultiHeadedAttention(h, d_model), 
                    feed_forward=PositionwiseFeedForward(d_model, d_ff, dropout)),
                PositionalEmbedding(
                    src_vocab, 
                    d_model,
                    dropout),
                PositionalEmbedding(
                    tgt_vocab, 
                    d_model,
                    dropout),
                Generator(tgt_vocab)
            )
    log.debug(f'model set up')
    return model