"""
Creates a model and trains it based on your parameters.

Created Aug 24 17:54:18 2023

@author: lmienhardt, lkuehl
"""

# tensorflow modules
import tensorflow as tf

# import from other documents
from training.ModelTrainer import ModelTrainer
from preprocessing.StoryTokenizer import StoryTokenizer
from settings.reserved_tokens import reserved_tokens
from settings.file_paths import train_file_path
from settings.file_paths import val_file_path
from settings.file_paths import vocab_path
from datapipeline.DatasetGenerator import DatasetGenerator

model_trainer = ModelTrainer(StoryTokenizer(reserved_tokens, vocab_path),
                                DatasetGenerator,
                                train_file_path,
                                val_file_path,
                                n_train_files=1478,
                                n_val_files=160,
                                dataset_lines_per_file=10000,
                                train_val_test_size=(1,1,0),
                                d_model=1024,
                                n_stacks=6,
                                h_att=8,
                                max_padding=128,
                                global_batch_size=64,
                                warmup_steps=10000,
                                n_epochs=4,
                                initial_epoch=0,
                                verbosity=1,
                                distributed_strategy=tf.distribute.MultiWorkerMirroredStrategy(),
                                load_model=False,
                                save_model=True,
                                save_keras_model=False,
                                model_load_path=None)
    
model_trainer.run_model()