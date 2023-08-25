"""
A mixin-Class for the tensorflow layers that enable visualization during non-training sessions.

Created Aug 25 12:59:02 2023

@author: lmienhardt, lkuehl
"""

# logging and decorators
import logging as log

# general modules
import numpy as np

# tensorflow modules
import tensorflow as tf
from tensorflow.python.keras.utils import plot_model

# necessary for visualization and user input
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import ipywidgets as widgets
from ipywidgets import VBox, HTML
from IPython.display import display

# import from other documents
from utilities.do_nothing import do_nothing

class VisualWrapper():
    """This is a mixin-Class for the tensorflow layers that enable visualization during non-training sessions."""
    instances = []          # save instances of VisualWrapper for reset_counter classmethod (see below)
    n_vis_layers_per_class = {
        'StoryTokenizer': 1,
        'EncoderDecoder': 1,
        'MultiHeadedAttention': 1,
        'StoryTokenizer': 1,
        'PositionalEmbedding': 1,
        'Generator': 1,
        'ResidualSublayer': 1,
    }
    vis_data = []

    def __init__(self, vis_on_count=None):
        """
        Initialize a VisualWrapper instance.

        Args:
            vis_on_count (list, optional):  A list of counts on which to perform a visualizations. 
                                            If not provided, no operations will be performed on any count.
            enabler (bool, optional):       A flag used to control whether visualization is enabled. 
                                            If False, it ensures no child class does perform any visualization.
                                            Defaults to False.

        The initialized instance is appended to the `VisualWrapper.instances` list, 
        the reset_counter classmethod resets the counters of all instances in the list.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        self.counter = 0
        self.vis_on_count = vis_on_count if vis_on_count else [0]
        if type(self).__name__ in self.n_vis_layers_per_class:
            num_instances = sum(isinstance(obj, type(self)) for obj in VisualWrapper.instances)
            if num_instances < self.n_vis_layers_per_class[type(self).__name__]:
                log.debug(f'append {self} to VisualWrapper.instances')
                VisualWrapper.instances.append(self)

    def increase_count(self):
        """Increase counter"""
        log.debug(f'execute')
        self.counter += 1

    # TODO: Enter standard texts and labels.
    def save_data(self, 
                  text, 
                  x, 
                  mode_x, 
                  text_x,
                  y=None, 
                  z=None,
                  mode_y=None,  
                  mode_z=None,
                  text_y=None,
                  text_z=None, 
                  x_axis=None, 
                  y_axis=None):
        """Saving data for visualization"""
        log.debug(f'execute')
        if self in self.instances:
            if self.counter in self.vis_on_count:
                self.increase_count()
                log.debug(f'append data to vis_data')
                self.vis_data.append({'x': x, 
                                    'y': y,
                                    'z': z,
                                    'mode_x': mode_x,
                                    'mode_y': mode_y,
                                    'mode_z': mode_z,
                                    'text': text,
                                    'text_x': text_x,
                                    'text_y': text_y,
                                    'text_z': text_z,
                                    'x_axis': x_axis,
                                    'y_axis': y_axis})

    @classmethod         
    def vis_data_generator(cls):
        log.debug(f'initialize generator')
        for data in cls.vis_data:
            log.debug(f'yield {data}')
            yield data

    @classmethod
    def visualize_data(cls):
        """
        Visualizes data_x (data_y) if tensorflow is not in training mode and global self.shoule_visualize is True.
        This only happens while self.counter is in self.vis_on_count (counter is increased by 1 with every execution)

        Args:
            data_x (unspecific often tensors):              The data that is visualized. TODO: Implement type check, to catch errors in advance
            mode (string):                                  One of the modes available in choose_func (see methods) to select the visualisation format.
            training (bool):                                Boolean parameter used by tensorflow to differentiate training and inference.
                                                            Only visualize when not in training mode.
            text (string):                                  Explanatory text giving information about the visualisation data.
                                                            Printed before visualisation is displayed.
            data_y (unspecific often tensors, optional):    TODO: Implement for multiple data visualization
            vis_diff (bool, optional):                      TODO: Implement for multiple data visualisation
        """
        log.debug(f'execute')
        
        vis_data_gen = cls.vis_data_generator()

        button = widgets.Button(description="Click to proceed")
        output = widgets.Output()

        def on_button_clicked(b):
            log.debug(f'execute')
            with output:
                # if all checks for visualization are passed execute visualisation

                try:
                    data = next(vis_data_gen)
                    text = data['text']
                    display_values = []
                    display_values.append((data['x'], data['mode_x'], data['text_x']))
                    display_values.append((data['y'], data['mode_y'], data['text_y']))
                    display_values.append((data['z'], data['mode_z'], data['text_z']))
                    
                    log.debug(f'visualise data: {data}')
        
                    # print explanatory text
                    cls.display_text(text)

                    for data in display_values:
                        # choose the correct visualization function
                        visualisation_func = cls.choose_func(data[1])
                        # print explanatory text
                        cls.display_text(data[2])
                        # apply visualization function to data_x
                        visualisation_func(data[0])
                except StopIteration:
                    log.debug(f'Vis data generator exhausted.')
                    b.disabled = True

        button.on_click(on_button_clicked)
        box = VBox([output, button])
        display(box)

    @classmethod    
    def choose_func(cls, mode):
        """
        This function returns an executable function for the chosen 'mode'.

        Args:
            mode (string): The string indicating the visualization mode to apply.

        Returns:
            function: An executable function taking one input argument. This argument should be the data to be visualized.
        """
        log.debug(f'execute')
        if mode == 'color_bar':
            return lambda x: cls.color_bar(x)
        elif mode == 'print':
            return lambda x: cls.display_text(x)
        elif mode == 'reduce_dim':
            return lambda x: cls.reduce_dim(x)
        elif mode == 'matrix':
            return lambda x: cls.matrix_repr(x)
        else:
            # return a placeholder function, if no valid 'mode' is given.
            return do_nothing

    @classmethod
    def display_text(cls, text):
        log.debug(f'execute')
        if isinstance(text, str):
            display(HTML('<p style="font-size:18px; color:blue;">' + text + '</p>'))

    @classmethod
    def color_bar(cls, tensor, xlabel=None, ylabel=None):
        """
        Use matplotlib to plot a colorbar that visualizes the values of a 1-D-tensor.

        Args:
            tensor (tf.tensor): The tensor to be visualized
        """ 
        log.debug(f'execute')
        # labels for the plot TODO: Generalize such that the labels are valid for all data types.
        x_label = xlabel or 'Tiefe'
        y_label = xlabel or 'Position'

        # Assuming data[0] is a numpy array.
        # If it's a ListWrapper or another list-like object, convert it to a numpy array.
        # TODO: Doesn't work. Check for error.
        data_array = np.array(tf.squeeze(tensor))

        # If the array is 1D, reshape it into a 2D array with one column
        if data_array.ndim != 2:
            log.error('Error: Expected a 1D tensor')
            return

        # Set the size of the plot (you can adjust the dimensions as needed)
        fig, ax = plt.subplots(figsize=(10, 2))

        # Use matshow to create a color-coded visualization
        cax = ax.matshow(data_array, cmap='jet', aspect='auto')

        # Add colorbar
        fig.colorbar(cax, label='Wertebereich')

        # Set labels
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # Set x and y tick locations to the middle of the cells
        #ax.set_xticks(np.arange(data_array.shape[1]), minor=False)
        #ax.set_yticks(np.arange(data_array.shape[0]), minor=False)

        plt.show()

    @classmethod
    def matrix_repr(cls, matrix):
        log.debug(f'execute')
        matrix = np.array(tf.squeeze(matrix))

        # If the tensor is not 3D, print an error message and return
        if matrix.ndim != 3:
            log.error('Error: Expected a 3D tensor')
            return

        # Calculate the number of subplots
        n_plots = matrix.shape[0]

        # Define the subplot grid dimensions (trying to get a roughly square grid)
        n_rows = int(np.sqrt(n_plots))
        n_cols = n_plots // n_rows if n_plots % n_rows == 0 else n_plots // n_rows + 1

        # Create a figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

        # Flatten axes for easier iteration
        axes = axes.flatten()

        # Iterate over each matrix in the tensor
        for i in range(n_plots):
            # Create a color-coded visualization of the matrix
            im = axes[i].imshow(matrix[i, :, :], cmap='jet')
            axes[i].set_xlabel('Eingabe')
            axes[i].set_ylabel('Ausgabe')

            # Set the title of the plot to indicate which matrix is being visualized
            axes[i].set_title(f'Aufmerksamkeitskopf {i + 1}')

        # Add colorbar, associating with the last image created
        fig.colorbar(im, ax=axes.ravel().tolist(), label='Wertebereich')

        plt.show()

    @classmethod
    def reduce_dim(cls, tensor):
        """
        Reduces the dimensionality of the input tensor using PCA and plots the result.

        This function first scales the input tensor by its minimum absolute value, then applies PCA to reduce its 
        dimensionality to 3. It then creates a 3D quiver plot of the reduced data.

        Args:
            tensor (np.ndarray): The input tensor to be reduced and visualized. 

        Shows:
            A 3D matplotlib plot of the tensor after dimensionality reduction using PCA.
        """
        log.debug(f'execute')
        # Reduce the first dimension, to create a 1-D numpy array.
        array = np.squeeze(tensor, axis=0)

        # Scale the array by its minimum absolute value to normalize the data
        scaled_array = array / np.min(np.abs(array))

        # Apply PCA for dimensionality reduction.
        # This reduces the dimensions of the data to 3.
        # TODO: PCA must be trained. Alternative algorithms could be tsne or umap.
        pca = PCA(n_components=3)
        reduced_array = pca.fit_transform(scaled_array)

        # Create a new figure and a set of subplots. 
        # The figure size is set to (3,3) to maintain a square aspect ratio. 
        # TODO: Find best size for plot
        fig, ax = plt.subplots(figsize=(3, 3))
        # Add another subplot to create a 3D plot.
        ax = fig.add_subplot(111, projection='3d')

        # Create a quiver plot to visualize each point as a vector from the origin
        ax.quiver(0, 0, 0, reduced_array[:, 0], reduced_array[:, 1], reduced_array[:, 2], arrow_length_ratio=0.1)

        # Label each component (PCA dimension) on the axes.
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        # Set a title for the plot
        # TODO: Generalize the title
        ax.set_title('Embeddings')

        # Set the plot boundaries to be the maximum value in the reduced array.
        boundaries = np.max(reduced_array)
        ax.set_xlim([-boundaries, boundaries])
        ax.set_ylim([-boundaries, boundaries])
        ax.set_zlim([-boundaries, boundaries])

        # Disply the plot
        plt.show()

    @classmethod
    def reset_visualiser(cls):
        """Reset the counter for all instances of the class."""
        log.debug(f'execute')
        for instance in cls.instances:
            instance.counter = 0
        cls.vis_data = []