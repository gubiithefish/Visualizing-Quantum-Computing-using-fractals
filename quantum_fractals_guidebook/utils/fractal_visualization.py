# Importing standard python libraries
from typing import Union
# from math import pi,sqrt
# from threading import *
from io import StringIO, BytesIO
# Import additional python libraries
# from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import clear_output, display, HTML
from celluloid import Camera
from PIL import Image
import base64
# from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.animation import ArtistAnimation

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit
# from qiskit.tools.jupyter import *
from qiskit.visualization import * # plot_bloch_multivector
import datetime
# from ipywidgets import Output
# out = Output()
from copy import deepcopy
# Define the animations class


class QuantumFractalVisualization:
    def __init__(self):
        # Variables for Figure and Axis for the Animation method
        self.gif_fig: Union[plt.subplots, None] = None
        self.gif_ax: Union[plt.subplots, None] = None
        self.camera: Union[Camera, None] = None

        # Shared data the bloch sphere to reduce compute time
        self.bloch_data: Union[BytesIO, None] = None
        self.iterations: Union[int, None] = None

    def save_bloch_as_obj(self, quantum_circuit, frame):
        """Instead of saving the bloch sphere as an image, the bloch sphere is saved as an in-memory object."""
        if self.iterations == frame:
            pass
        else:
            self.bloch_data = BytesIO()
            plot_bloch_multivector(quantum_circuit).savefig(self.bloch_data, format='png')
            self.bloch_data.seek(0)
            self.iterations = frame

    def qf_images(self, viz_data:Union[np.ndarray, QuantumCircuit], frame: int = 0) -> None:
        # Dynamically change the number of items included in the final output image
        fig, ax = plt.subplots(1, len(viz_data), figsize=(20, 5), clear=True)
        clear_output(wait=True)
        # Ensure each item in the visualization data is added to the figure
        for viz_idx, viz_obj in enumerate(viz_data):
            if isinstance(viz_obj, QuantumCircuit):
                # Save bloch sphere as an in-memory object.
                self.save_bloch_as_obj(quantum_circuit=viz_obj, frame=frame)

                # Insert the Bloch Sphere image data
                ax[viz_idx].imshow(Image.open(deepcopy(self.bloch_data)))

            if isinstance(viz_obj, np.ndarray):
                # Insert the calculated Julia Set data as an image
                ax[viz_idx].imshow(viz_obj, cmap='magma')

            # Turn off axis lines and labels
            ax[viz_idx].axis('off')

        # Finally show the image
        plt.show()
        plt.close()

    def qf_animation(self, viz_data: Union[np.ndarray, QuantumCircuit], frame: int = 0) -> None:
        if self.gif_fig is None and self.gif_ax is None:
            # Dynamically change the number of items included in the final output images
            self.gif_fig, self.gif_ax = plt.subplots(1, len(viz_data), figsize=(20, 5), clear=True)
            self.camera = Camera(self.gif_fig)
            clear_output(wait=True)

        # Ensure each item in the visualization data is added to the figure
        for viz_idx, viz_obj in enumerate(viz_data):
            if isinstance(viz_obj, QuantumCircuit):
                # Save bloch sphere as an in-memory object.
                self.save_bloch_as_obj(quantum_circuit=viz_obj, frame=frame)

                # Insert the Bloch Sphere image data
                self.gif_ax[viz_idx].imshow(Image.open(deepcopy(self.bloch_data)))

            if isinstance(viz_obj, np.ndarray):
                # Insert the calculated Julia Set data as an image
                self.gif_ax[viz_idx].imshow(viz_obj, cmap='magma')

            # Turn off axis lines and labels
            self.gif_ax[viz_idx].axis('off')

        # Finally take a snapshot of the image
        self.camera.snap()
        plt.close()

    def save_animation(self, blit:bool = True, interval_ms:int = 200, no_frames:int = 60, no_qubits: int = 1):
        anim = self.camera.animate(blit=blit, interval=interval_ms)
        anim.save(f'img/Quantum_Fractal_Animation_{no_frames}_frames_{no_qubits}_qubits.gif', writer='ffmpeg')

        clear_output(wait=True)
        with open(f"img/Quantum_Fractal_Animation_{no_frames}_frames_{no_qubits}_qubits.gif", 'rb') as fd:
            b64 = base64.b64encode(fd.read()).decode('ascii')
        return HTML(f'<img src="data:image/gif;base64,{b64}" />')

