"""
Defines a complete sequence generation model for a Transformer. 

Created Aug 25 13:05:12 2023

@author: lmienhardt, lkuehl
"""

# logging and decorators
import logging as log

# tensorflow modules
import tensorflow as tf
from tensorflow.python.keras.utils import plot_model

# necessary for visualization and user input
import ipywidgets as widgets
from IPython.display import display

# import from other documents
from utilities.VisualWrapper import VisualWrapper

class WordComplete(tf.Module, VisualWrapper):
  """
    This class defines a complete sequence generation model for a Transformer. 
    It uses a given tokenizer and Transformer model to generate sequences.
  """
  def __init__(self, 
               tokenizer, 
               transformer, 
               max_length=512, 
               pad_id=0,
               dtype=tf.Tensor, 
               decode_result=True):
    """
    Args:
        tokenizer (Tokenizer):          Tokenizer object to convert raw text into tokens.
        transformer (tf.keras.Model):   A Transformer model used for sequence generation.
        max_length (int, optional):     The maximum length of sequences that can be generated.
                                        Default is 512.
        dtype (tf.Tensor, optional):    The datatype of the output tensor. Default is tf.Tensor.
        decode_result (bool, optional): If True, decode the output tensor into a string. 
                                        Default is True.
    """
    log.debug(f'initialize {self.__class__.__name__}')
    super().__init__()
    VisualWrapper.__init__(self)
    self.tokenizer = tokenizer
    self.transformer = transformer
    self.max_length = max_length
    self.pad_id = pad_id
    self.dtype = dtype
    self.decode_result = decode_result
  
  def __call__(self, input, decode=True, encoding='utf-8', interactive=False):
    """
    Performs the sequence generation.

    Args:
        input (str or tf.Tensor):   The input sequence.
        decode (bool, optional):    If True, the output sequence is decoded into a string. 
                                    Default is True.
        encoding (str, optional):   The encoding to use when decoding the output sequence. 
                                    Default is 'utf-8'.
        training (bool, optional):  Whether the model is currently training. Default is None.

    Returns:
        text (str or tf.Tensor):    The generated text. If decode_result is True, this is a string.
                                    Otherwise, it is a tensor.
        tokens (tf.Tensor):         The tensor of generated tokens.
    """
    # during model set-up visualise data is created
    VisualWrapper.reset_visualiser()

    # initialize loading widget
    if interactive:
      load_bar = widgets.FloatProgress(value=0,
                                       min=0,
                                       max=self.max_length,
                                       description='Lädt',
                                       bar_style='info',
                                       style={'bar_color': 'green'},
                                       orientation='horizontal')
      display(load_bar)

    # TODO: Bug with empty strings as input
    # Convert input to tensor if it is not already
    # Create a dynamic tensor to store output
    # Make sure tensor_input is 2-D
    tensor_input = tf.convert_to_tensor(input)
    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    if len(tensor_input.shape) == 0:
      tensor_input = tensor_input[tf.newaxis]

    # tokenize and encode input
    # Identify end token of the input
    tokenized_input = self.tokenizer.tokenize(tensor_input).to_tensor()
    input_without_eos = tokenized_input[:, :-1]
    context = self.transformer.encode(input_without_eos, None)
    end = tokenized_input[-1][-1]

    # Write the input tokens (excluding the last one) to the output array
    for i, value in enumerate(tokenized_input[0][:-1]):
      output_array = output_array.write(i, value)

    # Start the generation of sequence from the last position of the input to max_length
    for i in tf.range(output_array.size(), self.max_length):
    
      if interactive:
        load_bar.value=i

      # Prepare input for decoder
      # Decode the input
      dec_input = output_array.concat()[tf.newaxis]

      decode = self.transformer.decode(context, None, dec_input, None)

      # Create logits predictions and select the last predicted token
      predictions = self.transformer.generator(decode)
      predictions_last = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.
      predicted_id = tf.argmax(predictions_last, axis=-1)

      # Concatenate the `predicted_id` to the output which is given to the decoder as its input again.
      output_array = output_array.write(i, predicted_id[0][0])

      # break the loop, if [End] token is predicted
      if predicted_id == end:
        break
    
    if interactive:
      load_bar.value = load_bar.max
    # Create a tensor for detokenization
    # Detokenize
    # Create tokens from detokenized output again
    output = output_array.concat()[tf.newaxis]
    text = self.tokenizer.detokenize(output)
    tokens = self.tokenizer.lookup(output)

    # If decode_result is True, decode the text tensor into a string
    if self.decode_result:
      text = text.numpy()[0].decode(encoding)
      print(text)
  
  def print_results(self, visualisation=False):
    if visualisation:
      VisualWrapper.visualize_data()