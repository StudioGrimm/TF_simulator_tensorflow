"""
Creates embedding. 

Created Aug 25 13:35:41 2023

@author: lmienhardt, lkuehl
"""

# tensorflow modules
import tensorflow as tf

# necessary for visualization and user input
import ipywidgets as widgets

# import from other documents
from preprocessing.StoryTokenizer import StoryTokenizer
from settings.reserved_tokens import reserved_tokens
from settings.file_paths import vocab_path
from utilities.VisualWrapper import VisualWrapper
from transformermodel.ModelLoader import ModelLoader

# TODO: put the 'model' as an input argument
model = ModelLoader(StoryTokenizer(reserved_tokens, vocab_path),
                            d_model=512,
                            n_stacks=6,
                            h_att=8,
                            load_model=True,
                            model_load_path="model_N6_h8_d512_t20230614-075226")

class EmbeddingExample():

    def __init__(self) -> None:
        self.tokenizer = StoryTokenizer(reserved_tokens, vocab_path)

        self.input_widget = widgets.Text(
            value = 'Einbettung Test',
            description = 'Ihre Eingabe:',
            continuous_update=False,  # updates value only when you finish typing or hit "Enter"
            layout = widgets.Layout(width='auto', margin='0px 0px 10px 0px')
        )

        self.button_widget = widgets.Button(description='Einbettung erstellen',
                                    layout = widgets.Layout(width='auto'))

        self.output_widget = widgets.Output(layout = widgets.Layout(width='auto'))
        self.old_context = None

    def on_button_click(self, b):
        with self.output_widget:
            self.output_widget.clear_output()  # clear the previous output
            VisualWrapper.reset_visualiser()
            tokens = self.tokenizer.tokenize(self.input_widget.value)
            input_without_eos = tokens[tf.newaxis, :, :-1]
            context = model.model.enc_embed(input_without_eos)
            VisualWrapper.display_text('So sieht die Einbettung der Eingabe aus.')
            VisualWrapper.color_bar(context.to_tensor())

            if self.old_context is not None:
                padded_context, padded_old_context = self.pad_tensors(context, self.old_context)

                VisualWrapper.display_text('So unterscheiden sich die alte und die neue Einbettung voneinander.')
                context_diff = padded_context - padded_old_context
                VisualWrapper.color_bar(context_diff)

            self.old_context = context
    
    def pad_tensors(self, ragged_tensor1, ragged_tensor2):
        # Convert ragged tensors to normal tensors, padding with zeros
        tensor1 = ragged_tensor1.to_tensor()
        tensor2 = ragged_tensor2.to_tensor()

        # Calculate the shapes of the tensors
        shape1 = tf.shape(tensor1)
        shape2 = tf.shape(tensor2)

        # Initialize a list for the target shape
        target_shape = []

        # Iterate over the dimensions of the tensors
        for i in range(shape1.shape[0]):
            # Append the maximum size of the dimension to the target shape
            target_shape.append(tf.maximum(shape1[i], shape2[i]))

        # Convert the target shape to a tensor
        target_shape = tf.stack(target_shape)

        # Initialize lists for the paddings of the tensors
        paddings1 = []
        paddings2 = []

        # Iterate over the dimensions of the tensors
        for i in range(shape1.shape[0]):
            # Append the required padding for the dimension to the paddings
            paddings1.append([0, target_shape[i] - shape1[i]])
            paddings2.append([0, target_shape[i] - shape2[i]])

        # Convert the paddings to tensors
        paddings1 = tf.stack(paddings1)
        paddings2 = tf.stack(paddings2)

        # Pad the tensors to the target shape
        tensor1_padded = tf.pad(tensor1, paddings1)
        tensor2_padded = tf.pad(tensor2, paddings2)

        return tensor1_padded, tensor2_padded