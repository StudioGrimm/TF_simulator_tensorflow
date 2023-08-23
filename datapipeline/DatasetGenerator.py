"""
A class to generate TensorFlow datasets for Transformer models from text files.

Created Aug 23 17:19:59 2023

@author: lmienhardt, lkuehl
"""

# logging and decorators
import logging as log

# system tools
import pathlib

# general modules
import numpy as np
import itertools

# tensorflow modules
import tensorflow as tf

# import from other files
from utilities.subsequent_mask import subsequent_mask

class DatasetGenerator():
    """
    A class to generate TensorFlow datasets for Transformer models from text files.
    The txt_files_to_lines_gen generates lines, that are fitted into a certain length by
    lines_to_fit_sentences (it combines follow-up sentences, if they don't exceed the limit together)..
    generate_datasets and prepare_datapoint are used to generate the kind of data necessary for our model:
    (src, tgt, src_mask, tgt_mask), label. Here src, tgt and label are similar tensors, but shifted right or left
    and with or without [Start] or [End] tokens. The src_mask is a padding mask and the tgt_mask is a subsequent mask.
    """

    def __init__(self,
                 tokenizer,
                 file_path = None,
                 buffer_size=20000, 
                 batch_size=64,
                 max_padding=512, 
                 pad_id=0):
        """
        Constructor for the DatasetGenerator class.
        
        Args:
            tokenizer:          Instance of the tokenizer to be used.
            buffer_size (int):  Number of elements from the dataset from which the new dataset will sample.
                                This is crucial for randomisation, as the dataset is not shuffled further than buffer_size allows.
            batch_size (int):   Number of elements per batch in the dataset.
            max_padding (int):  The maximum sequence length, shorter sequences will be padded with pad_id.
            pad_id (int):       ID to be used for padding shorter sequences.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        self.tokenizer = tokenizer
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.max_padding = max_padding
        self.pad_id = pad_id
        self.file_iter = pathlib.Path(file_path).iterdir()

    def txt_files_to_lines_gen(self, iterations):
        """
        Generator function that yields lines from text files in a directory.

        Args:
            file_path (str):    Path to the directory containing the text files.

        Yields:
            str:                A line from a text file.
        """
        log.debug(f'execute')
        log.debug(f'extract {iterations} files from dataset')
        # TODO There are more elegant solutions to this!
        for file in itertools.islice(self.file_iter, iterations):
            if file.is_file():
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        yield line.strip()
    
    def generate_dataset(self, n_files):
        """
        Generates a tokenized, batched TensorFlow dataset from text files.

        Args:
            file_path (str): Path to the directory containing the text files.

        Returns:
            tf.data.Dataset: The generated dataset. It contains the following data: (src, tgt, src_mask, tgt_mask), label
        """
        log.debug(f'execute')

        # Create a Dataset from the text file
        lines_gen = self.txt_files_to_lines_gen(n_files)
        log.debug(f'generators set up')
        
        dataset = tf.data.Dataset.from_generator(lambda: lines_gen, 
                                                 output_signature=tf.TensorSpec(shape=(), 
                                                                                dtype=tf.string))
        log.debug(f'dataset created')

        # Tokenize the whole dataset with the pre-trained tokenizer and apply our data preparation method.
        prepared_data_set = (dataset
                             .shuffle(self.buffer_size)
                             .batch(self.batch_size)
                             .map(lambda x: self.prepare_datapoint(x), 
                                num_parallel_calls=tf.data.AUTOTUNE)
                             .prefetch(buffer_size=tf.data.AUTOTUNE))

        log.debug(f'dataset processed')
        
        return prepared_data_set

    def prepare_datapoint(self, batch):
        """
        Prepares a datapoint for the transformer model by tokenizing and creating the necessary masks.

        Args:
            data_point (str): A sentence or text to be prepared.

        Returns:
            tuple: A tuple containing source tokens, target tokens and their respective masks, and label tokens.
        """
        log.debug(f'execute')
        src_tokens = self.tokenizer.tokenize(batch)
        # Shorten tgt and label in order to remove [Start], [End] tokens
        tgt_tokens = src_tokens[:, :-1]
        label_tokens = src_tokens[:, 1:]
        
        # Fill the data to same size tensors.
        src = src_tokens.to_tensor(shape=[None, self.max_padding], 
                                   default_value=self.pad_id)
        tgt = tgt_tokens.to_tensor(shape=[None, self.max_padding], 
                                   default_value=self.pad_id)
        label = label_tokens.to_tensor(shape=[None, self.max_padding], 
                                       default_value=self.pad_id)
        
        # padding mask for source and subsequent mask for tgt
        # the masks are to be passed with the data through the model instead of reacreating it every time the model runs.
        src_mask = (src != self.pad_id)[:, np.newaxis, :]
        tgt_mask = self.make_subseq_mask(tgt)

        return (src, tgt, src_mask, tgt_mask), label
  
    def make_subseq_mask(self, tgt):
        """
        Creates a mask for the transformer model to avoid using future tokens and padding.

        Args:
            tgt (tf.Tensor): Tensor of target tokens.

        Returns:
            tf.Tensor: The mask tensor.
        """
        log.debug(f'execute')
        tgt_mask = (tgt != self.pad_id)[:, np.newaxis, :]
        tgt_mask = tf.logical_and(tgt_mask, subsequent_mask(tgt.shape[-1]))
        return tgt_mask