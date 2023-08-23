"""
Performs tokenization with the BERT tokenizer.

Created Aug 23 17:12:33 2023

@author: lmienhardt, lkuehl
"""

# logging and decorators
import logging as log

# system tools
import pathlib

# tensorflow modules
import tensorflow as tf
import tensorflow_text as tf_text

# import from other files
from settings.reserved_tokens import reserved_tokens

class StoryTokenizer(tf.Module):
    """
    The StoryTokenizer class is designed to perform tokenization and detokenization tasks using the BERT tokenizer.
    
    Methods:
        tokenize:               Tokenize a string with BERT Tokenizer, add [Start] and [End] tokens.
        detokenize:             Detokenize a token vector, clean the string of the reserved tokens.
        lookup:                 Return the tokens a string is composed of.
        add_start_end:          Add [Start], [End] toknes to a raggend token vector.
        cleanup_text:           Remove reserved tokens from a string.
        get_vocab_size:         Return the length of the vocabulary used by the tokenizer.
        get_vocab_path:         Return the path of the vocabulary filee.
        get_reserved_tokens:    Return a list of all reserved tokens.
    """
    def __init__(self, reserved_tokens, vocab_path):    
        """
        Initialize a StoryTokenizer

        Args:
            reserved_tokens (list of strings):  A list of strings with special tokens
            vocab_path (string):                The path to the vocabulary file
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()

        self.tokenizer = tf_text.BertTokenizer(vocab_path, lower_case=True)
        self._reserved_tokens = reserved_tokens
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        # read in the vocabulary from file.
        vocab = pathlib.Path(vocab_path).read_text(encoding='utf-8').splitlines()
        self.vocab = tf.Variable(vocab)        

    def tokenize(self, strings, training=None):
        """
        Tokenizes the input strings and adds start and end tokens.

        Args:
            strings (tf.Tensor):        The strings to be tokenized.
            training (bool, optional):  If True, the model is in training mode. Defaults to None.

        Returns:
            out (tf.RaggedTensor):      The tokenized strings with added start and end tokens.
        """
        log.debug(f'execute')
        encoded = self.tokenizer.tokenize(strings)
        merged_enc = encoded.merge_dims(-2, -1)
        out = self.add_start_end(merged_enc)

        return out
    
    def detokenize(self, tokenized, training=None):
        """
        Detokenizes the input token IDs back into text strings.
        Any reserved tokens (except for "[UNK]") are removed from the detokenized text.

        Args:
            tokenized (tf.RaggedTensor): The token IDs to be detokenized.
            training (bool, optional): If True, the model is in training mode. Defaults to None.

        Returns:
            tf.Tensor: The detokenized text.
        """
        log.debug(f'execute')
        words = self.tokenizer.detokenize(tokenized)
        return self.cleanup_text(self._reserved_tokens, words)
    
    def lookup(self, token_ids):
        """
        Converts token IDs to their corresponding token strings from the vocabulary.

        Args:
            token_ids (tf.RaggedTensor or tf.Tensor): The token IDs to be converted.

        Returns:
            tf.RaggedTensor or tf.Tensor: The corresponding token strings.
        """
        log.debug(f'execute')
        return tf.gather(self.vocab, token_ids)

    @staticmethod
    def add_start_end(ragged):
        """
        Adds start and end tokens to the input token IDs.

        Args:
            ragged (tf.RaggedTensor): The input token IDs.

        Returns:
            tf.RaggedTensor: The token IDs with added start and end tokens.
        """
        log.debug(f'execute')
        # Create vectores for the [Start] and [End] tokens.
        START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
        END = tf.argmax(tf.constant(reserved_tokens) == "[END]")

        # fill up dim 0 and concat in dim 1 to handle batches.
        count = ragged.bounding_shape()[0]
        starts = tf.fill([count, 1], START)
        ends = tf.fill([count, 1], END)
        return tf.concat([starts, ragged, ends], axis=1)

    @staticmethod
    def cleanup_text(reserved_tokens, token_txt):
        """
        Removes any reserved tokens (except for "[UNK]") from the input text.

        Args:
            reserved_tokens (list of str): The list of reserved tokens.
            token_txt (tf.Tensor): The input text.

        Returns:
            tf.Tensor: The cleaned up text.
        """
        log.debug(f'execute')
        # Create a regular expression searching for reserved tokens
        bad_tokens = list(filter(lambda token: token != "[UNK]", reserved_tokens))
        bad_tokens_re = "|".join(bad_tokens)

        # Search and delete reserved tokens from the token_txt tensor
        bad_cells = tf.strings.regex_full_match(token_txt, bad_tokens_re)
        ragged_result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

        # join the text
        result = tf.strings.reduce_join(ragged_result, separator=' ', axis=-1)

        return result
    
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]
    
    def get_vocab_path(self):
        return self._vocab_path
    
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)