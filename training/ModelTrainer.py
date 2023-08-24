"""
Trains or loads the model.

Created Aug 24 17:26:39 2023

@author: lmienhardt, lkuehl
"""

# logging and decorators
import logging as log
import time

# system tools
import pathlib

# general modules
import numpy as np
import math

# tensorflow modules
import tensorflow as tf

# import from other documents
from accuracy import accuracy_with_pad_idx
from LabelSmoothingLoss import LabelSmoothingLoss
from LossCompute import LossCompute
from TransformerSchedule import TransformerSchedule
from transformermodel.make_model import make_model


class ModelTrainer():
    """
    Documentation
    """
    def __init__(self, 
                 tokenizer, 
                 data_generator,
                 train_path,
                 val_path,
                 dataset_lines_per_file = None,
                 n_train_files = 0,
                 n_val_files=0,
                 train_val_test_size = (0, 0, 0),
                 d_model = 512,
                 n_stacks = 6,
                 h_att = 8,
                 smoothing = 0.1,
                 max_padding = 512,
                 pad_idx = 0,
                 global_batch_size = 64,
                 warmup_steps = 4000,
                 base_lr = None,
                 n_epochs = 1,
                 initial_epoch = None,
                 verbosity = 2,
                 distributed_strategy = tf.distribute.MirroredStrategy(),
                 load_model = False,
                 save_model = True,
                 save_keras_model = False, # saves the whole model, not just the weights
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
        self.smoothing = smoothing
        self.accuracy = accuracy_with_pad_idx(pad_idx)

        # var for distributed training
        self.strategy = distributed_strategy
        self.n_devices = distributed_strategy.num_replicas_in_sync
        log.debug(f"number of processing devices = {self.n_devices}")

        # var for batching
        self.global_batch_size = global_batch_size
        self.per_device_batch_size = (global_batch_size / self.n_devices)

        # var for data generation
        self.n_dataset_files = n_train_files
        self.lines_per_file = dataset_lines_per_file
        self.train_val_test_size = np.array(train_val_test_size)
        self.data_path = train_path
        self.train_n_files = np.floor(self.train_val_test_size[0] * n_train_files).astype(int)
        self.val_n_files = np.floor(self.train_val_test_size[1] * n_val_files).astype(int)
        self.max_padding = max_padding
        self.pad_idx = pad_idx
        self.train_generator = data_generator(tokenizer = tokenizer, 
                                             file_path = train_path,
                                             batch_size = self.global_batch_size,
                                             max_padding = max_padding,
                                             pad_id = pad_idx)
        
        self.val_generator = data_generator(tokenizer = tokenizer, 
                                             file_path = val_path,
                                             batch_size = self.global_batch_size,
                                             max_padding = max_padding,
                                             pad_id = pad_idx)
        
        self.train_data = self.load_dataset(self.train_generator, self.train_n_files)        
        self.val_data = self.load_dataset(self.val_generator, self.val_n_files)
        
        #if self.strategy is not None:
        #    multi_dev_data_gen = lambda x: self.data_generator.multi_device_generate_dataset(x, data_path,
        #                                           train_val_test_size)
        #    self.train_data = self.strategy.distribute_datasets_from_function(multi_dev_data_gen)
        #    self.val_data = self.train_data # TODO Change this !!!!!
        #else:    
            
        # var for model fit
        self.n_epochs = n_epochs
        self.initial_epoch = 0 or initial_epoch
        self.data_steps_total = math.floor(self.train_n_files * dataset_lines_per_file / self.global_batch_size)
        self.validation_steps_total = math.floor(self.val_n_files * dataset_lines_per_file / self.global_batch_size)
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.fit_verbosity = verbosity
        self.callbacks = []

        
        # var for load and save
        self.load_model = load_model
        self.save_model = save_model
        self.save_keras_model = save_keras_model
        self.model_load_path = model_load_path

        # compile model and load model weights if applicable
        self.model = self.compile_model()
        if load_model:
            self.load_model_weights(self.model, self.d_model, self.model_load_path)
        
        # add checkpoints
        if save_model:
            self.add_save_checkpoints()

    def load_dataset(self, data_generator, file_numbers):
        """
        This function loads the dataset into the ModelTrainer.
        """
        log.debug(f'execute')
        return data_generator.generate_dataset(file_numbers)
        
    def compile_model(self):
        with self.strategy.scope():
            # set_up_model
            model = make_model(self.vocab_size, 
                               self.vocab_size, 
                               d_model = self.d_model,
                               N = self.n_stacks,
                               h = self.h_att)
            log.debug(f'model set up')

            # compile model
            model.compile(
            loss = LossCompute(model.generator, 
                               LabelSmoothingLoss(self.vocab_size, 
                                                  self.pad_idx, 
                                                  self.smoothing), 
                               self.vocab_size), 
            optimizer = tf.keras.optimizers.Adam(TransformerSchedule(self.d_model, 
                                                                     self.warmup_steps), # type: ignore
                                                                     beta_1=0.9, 
                                                                     beta_2=0.999, 
                                                                     epsilon=1e-9), 
            metrics = [self.accuracy],
            )
            log.debug(f'model compiled')

        return model
    
    def run_model(self, training_data=None, validation_data=None, epochs=None):
        """
        Execute model training
        """
        training_data = training_data or self.train_data
        validation_data = validation_data or self.val_data
        epochs = epochs or self.n_epochs
        n_devices = self.strategy.num_replicas_in_sync
        steps_per_epoch = math.floor(self.data_steps_total / (n_devices * 2))
        validation_steps = math.floor(self.validation_steps_total / (n_devices * 2))

        if training_data is None or validation_data is None or epochs is None:
            raise ValueError("Training data, validation data and epochs must be provided either as arguments or as instance attributes.")

        self.model.fit(training_data, 
                       epochs = epochs,
                       steps_per_epoch = steps_per_epoch,
                       validation_data = validation_data,
                       validation_steps = validation_steps,
                       callbacks = self.callbacks,
                       verbose = self.fit_verbosity)
        
        if self.save_model:
            self.save_model_weights()

        # if self.save_keras_model:
        #     #self.model.save('my_model', save_format="tf")
        #     tf.keras.saving.save_model(self.model, 'my_model')
        
        print(self.model.summary())
    
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
            directories = [model_folder]
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
                latest_epoch_file = h5_files[-1]
                latest_weights = latest_epoch_file

        log.debug(f'model weights extracted')

        # Load weights if we found a previously trained model
        if latest_weights is not None:
            print(f'Loading weights from {latest_weights}')
            
            # Create a dummy input matching the input shape of the model
            # TODO: Ensure that the shape and type of the dummy_input match with the actual input that your model is going to receive.
            dummy_input = tf.random.uniform(shape=[1,d_model]), tf.random.uniform(shape=[1,d_model]), None, None
            # Call the model on the dummy input
            _ = model.generator(model(dummy_input))

            model.load_weights(latest_weights)
        log.debug(f'model loaded with weights')

    def add_save_checkpoints(self):
        log.debug(f'execute')

        current_time = time.strftime("%Y%m%d-%H%M%S")

        directory = f"model_N{self.n_stacks}_h{self.h_att}_d{self.d_model}_t{current_time}"
        ckp_name = "model_{epoch:03d}.h5"
        dir_path = pathlib.Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        checkpoint_path = dir_path / ckp_name
        
        epoch_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                              save_freq='epoch',
                                                              save_weights_only=True, 
                                                              verbose=1)
        
        self.callbacks.append(epoch_checkpoint)
    
    def save_model_weights(self):
        log.debug(f'execute')

        # TODO: Ensure that checkpoints and final model are saved in the same folder.
        current_time = time.strftime("%Y%m%d-%H%M%S")

        directory = f"model_N{self.n_stacks}_h{self.h_att}_d{self.d_model}_t{current_time}"
        final_name = "final_model.h5"
        dir_path = pathlib.Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        save_path = dir_path / final_name

        self.model.save_weights(save_path, overwrite=True)