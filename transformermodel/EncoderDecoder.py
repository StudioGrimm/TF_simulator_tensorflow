"""
Defines a Transformer model for sequence-to-sequence tasks..

Created Aug 24 13:38:20 2023

@author: lmienhardt, lkuehl
"""

# logging and decorators
import logging as log

# tensorflow modules
import tensorflow as tf

class EncoderDecoder(tf.keras.Model):
    """
    Defines a Transformer model for sequence-to-sequence tasks.

    This class implements the Transformer architecture, which consists of an Encoder and Decoder, each built from multiple stacked self-attention and feedforward layers.
    Inherits from both the TensorFlow Keras Model class for building ML models and a custom VisualWrapper class for data visualization.

    Attributes:
        encoder_stack (Encoder):                The encoder component of the Transformer.
        decoder_stack (Decoder):                The decoder component of the Transformer.
        enc_embed (tf.keras.layers.Embedding):  The input embedding layer for the encoder.
        dec_embed (tf.keras.layers.Embedding):  The input embedding layer for the decoder.
        generator (tf.keras.layers.Dense):      The output linear layer.

    Note: We use two seperate embeddings, because the encoder get's the data with start token, while the decoder get's the data without start token.
    Note: To receive actual output from the model it is necessary to run call on the input and then the generator on the output.
    """

    def __init__(self, encoder_stack, decoder_stack, enc_embed, dec_embed, generator):
        """
        Initialize the EncoderDecoder model together with the parent VisualWrapper.
        The EncoderDecoder model is an visualization enabler, that means, that it enables visualization for all its sublayer, if self.counter in self.vis_on_count.

        Args:
            encoder_stack (layers.Layer):   A Encoder object, consisting of a stack of self-attention and feedforward layers.
                                            The stack size is determined within the object.
            decoder_stack (layers.Layer):   A Decoder object, consisting of a stack of self-attention, source-attention and feedforward layers.
                                            The stack size is determined within the object.
            enc_embed (layers.Layer):       An embedding layer for the encoder input.
            dec_embed (layers.Layer):       An embedding layer for the decoder input.
            generator (layers.Layer):       The final linear layer that generates predictions.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()

        self.encoder_stack = encoder_stack
        self.decoder_stack = decoder_stack
        self.enc_embed = enc_embed
        self.dec_embed = dec_embed
        self.generator = generator

    def encode(self, inputs, pad_mask, training=None):
        """
        Args:
            inputs (Tensor):            Input data tensor to encode.
            pad_mask (Tensor):          Mask tensor to ignore padding tokens in the input.
            training (bool, optional):  Boolean flag indicating whether the model is in training mode. Defaults to None.

        Returns:
            Tensor:                     The output of the encoder stack.
        """
        log.debug(f'execute')
        return self.encoder_stack(self.enc_embed(inputs), 
                                  pad_mask, 
                                  training=training)

    def decode(self, enc_input, pad_mask, inputs, subseq_mask, training=None):
        """
        Args:
            enc_input (Tensor):         Encoded input data tensor to decode.
            pad_mask (Tensor):          Mask tensor to ignore padding tokens in the input.
            inputs (Tensor):            Input data tensor for the decoder.
            subseq_mask (Tensor):       Mask tensor to ignore subsequent tokens in the input.
            training (bool, optional):  Boolean flag indicating whether the model is in training mode. Defaults to None.

        Returns:
            Tensor:                     The output of the decoder stack.
        """
        log.debug(f'execute')
        return self.decoder_stack(self.dec_embed(inputs), 
                                  enc_input, 
                                  pad_mask, 
                                  subseq_mask, 
                                  training=training)

    def call(self, inputs, training=None):
        """
        Args:
            inputs (tuple):             Tuple of Tensors (enc_input (Tensor, dtype=tf.float32), 
                                                          dec_input, (Tensor, dtype=tf.float32)
                                                          pad_mask, (Tensor, dtype=tf.bool)
                                                          subseq_mask(Tensor, dtype=tf.bool)
                                                         ).
            training (bool, optional):  Boolean flag indicating whether the model is in training mode. Defaults to None.

        Returns:
            Tensor: The output of the model.
        """
        # We need to unpack the input, as tensorflows model.fit method requires the input to be passed as a single parameter,
        # but it actually contains (enc_input, dec_input, pad_mask, subseq_mask) as a tuple.
        enc_input, dec_input, pad_mask, subseq_mask = inputs

        return self.decode(self.encode(enc_input, pad_mask, training), 
                           pad_mask,
                           dec_input, 
                           subseq_mask, training)