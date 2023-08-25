"""
Loads an already trained model (loads the weights into a model).

Created Aug 25 17:43:01 2023

@author: lmienhardt, lkuehl
"""

# logging and decorators
import logging as log

# system tools
import pathlib

# tensorflow modules
import tensorflow as tf

# import from other documents
from make_model import make_model
from utilities.VisualWrapper import VisualWrapper

class ModelLoader():
    """
    Documentation
    """
    def __init__(self, 
                 tokenizer,
                 d_model = 512,
                 n_stacks = 6,
                 h_att = 8,
                 load_model = False,
                 model_load_path = None
                 ):
        """
        Docstring
        """
        log.debug(f'initialize {self.__class__.__name__}')
        # class modules
        self.tokenizer = tokenizer

        # var for model compile
        self.vocab_size = tokenizer.get_vocab_size()
        self.d_model = d_model
        self.n_stacks = n_stacks
        self.h_att = h_att
        
        # var for load and save
        self.load_model = load_model
        self.model_load_path = model_load_path

        # compile model and load model weights if applicable
        self.model = self.set_up_model()
        if load_model:
            self.load_model_weights(self.model, self.d_model, self.model_load_path)
        
    def set_up_model(self):
        # set_up_model
        model = make_model(self.vocab_size, 
                            self.vocab_size, 
                            d_model = self.d_model,
                            N = self.n_stacks,
                            h = self.h_att)
        log.debug(f'model set up')

        VisualWrapper.reset_visualiser()

        return model

    def load_model_weights(self, model, d_model, model_folder):
        """
        Load the latest model weights if available.

        Args:
            model (tf.keras.Model):         The model to which the weights will be loaded.
            d_model (int):                  The dimension of the Transformer architecture.
            model_folder (str, optional):   The directory from which to load the weights. 
                                            Default is None.

        Returns:
            model (tf.keras.Model):         The model with the loaded weights.
            
        This function loads the weights from the latest trained model found in the provided model_folder 
        or from the latest model in the current directory if load_latest is True.
        """
        log.debug(f'execute')
        # TODO: Ensure architecture sizes match.
        if model_folder is not None:
            log.debug(f'model_folder={model_folder}')
            # Load weights from the specified model folder
            directories = [pathlib.Path(model_folder)]
        else:
            directories = sorted(pathlib.Path('.').glob('model_N*_h*'), key=lambda x: x.stat().st_mtime, reverse=True)

        log.debug(f'load_dir={directories}')

        # Load weights from the latest trained model
        latest_weights = None
        if directories:
            latest_dir_path = directories[0]
            # Get all the h5 files inside the directory and sort them
            h5_files = sorted(latest_dir_path.glob('*.h5'))

            if h5_files:
                # Pick the last epoch file (or final_model file if it exists)
                latest_weights = h5_files[-1]

        log.debug(f'model weights extracted')

        # Load weights if we found a previously trained model
        if latest_weights is not None:
            log.debug(f'Loading weights from {latest_weights}')
            
            # Create a dummy input matching the input shape of the model
            # TODO: Ensure that the shape and type of the dummy_input match with the actual input that your model is going to receive.
            dummy_input = tf.random.uniform(shape=[1,d_model]), tf.random.uniform(shape=[1,d_model]), None, None
            # Call the model on the dummy input
            _ = model.generator(model(dummy_input))

            model.load_weights(latest_weights)
        log.debug(f'model loaded with weights')