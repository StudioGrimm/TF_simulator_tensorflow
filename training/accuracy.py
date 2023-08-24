"""
Calculates masked_accuracy or accuracy_with_pad_idx.

Created Aug 24 17:29:09 2023

@author: lmienhardt, lkuehl
"""

# logging and decorators
import logging as log

# tensorflow modules
import tensorflow as tf

def masked_accuracy(label, pred, pad_idx):
  """
  This function calculates the accuracy of the prediction while ignoring the specified padding index. 
  It assumes that labels have already been converted into indices (i.e., not one-hot encoded).

  Args:
      label (tf.Tensor):  The ground truth labels. These should be integer indices, 
                          not one-hot encoded, with a shape of (batch_size, seq_length).
      pred (tf.Tensor):   The predicted labels, given by the model. These should be 
                          the raw outputs of the model (i.e., logits) with a shape of 
                          (batch_size, seq_length, vocab_size).
      pad_idx (int):      The index representing the padding in the sequence. This will be excluded 
                          from the accuracy calculation.

  Returns:
      tf.Tensor: The accuracy of the model's predictions, excluding padding. 
                  It is a scalar tensor (0-dimensional).
  """

  pred = tf.argmax(pred, axis=2)      # calculate prediction tokens from logits
  label = tf.cast(label, pred.dtype)  # assure matching works

  match = label == pred   # mask
  mask = label != pad_idx # mask out padding
  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)

  return tf.reduce_sum(match)/tf.reduce_sum(mask)

def accuracy_with_pad_idx(pad_idx):
   """
   Returns an accuracy function where the pad_idx is already set.add

   Args:
      pad_idx (int): An id for the padding token such that it can be masked in accuracy calculation

    Returns:
      function: A accuracy function that compares label and prediction.
   """
   return lambda label, pred: masked_accuracy(label, pred, pad_idx)
