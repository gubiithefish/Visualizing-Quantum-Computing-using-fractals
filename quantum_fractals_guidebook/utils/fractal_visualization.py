#!/usr/bin/env python
# coding: utf-8
# Credits & License: https://github.com/wmazin/Visualizing-Quantum-Computing-using-fractals

# Importing standard python libraries
from copy import deepcopy
from pathlib import Path
from typing import Union
from io import BytesIO
import base64

# Import additional python libraries
# -- Animation and visualization purposes
import matplotlib.pyplot as plt
from matplotlib import font_manager, animation
from IPython.display import clear_output, HTML
from celluloid import Camera
from PIL import Image

# -- Types
from numpy import ndarray

# -- Quantum
from qiskit.visualization import plot_bloch_multivector
from qiskit import QuantumCircuit

# Import project-modules
from .fractal_quantum_circuit import FractalQuantumCircuit
from .fractal_julia_calculations import set_1cn0, set_2cn1, set_2cn2

# Load fonts used for visualizations
# ───────────────────────────────────────────────────────────────────
font_path = str(Path(Path(__file__).resolve().parent.parent, "static", "fonts", "IBMPlexMono-SemiBold.ttf"))
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)


# ───────────────────────────────────────────────────────────────────
class QuantumFractalVisualization:
    def __init__(self):
        # Variables for Figure and Axis for the Animation method
        self.gif_fig: Union[plt.subplots, None] = None
        self.gif_ax: Union[plt.subplots, None] = None
        self.camera: Union[Camera, None] = None
        self.gif_gs: Union[int, None] = None

        # Shared data the bloch sphere to reduce compute time
        self.bloch_data: Union[BytesIO, None] = None
        self.iterations: Union[int, None] = None

    def save_bloch_as_obj(self, quantum_circuit: QuantumCircuit, frame:int = 0, return_obj: bool = False):
        """Instead of saving the bloch sphere as an image, the bloch sphere is saved as an in-memory object."""
        if return_obj is True:
            bloch_data = BytesIO()
            plot_bloch_multivector(quantum_circuit).savefig(bloch_data, format='png')
            bloch_data.seek(0)
            return bloch_data
        elif self.iterations == frame:
            pass
        else:
            self.bloch_data = BytesIO()
            plot_bloch_multivector(quantum_circuit).savefig(self.bloch_data, format='png')
            self.bloch_data.seek(0)
            self.iterations = frame

    def qf_images(self, viz_data:Union[ndarray, QuantumCircuit], frame: int = 0) -> None:
        plt.rcParams['font.family'] = prop.get_name()
        plt.rcParams["figure.figsize"] = (20, 5)
        plt.rcParams['figure.dpi'] = 60

        # Dynamically change the number of items included in the final output image
        img_fig, img_ax = plt.subplots(1, len(viz_data), figsize=(20, 5), clear=True)
        clear_output(wait=True)
        # Ensure each item in the visualization data is added to the figure
        for viz_idx, viz_obj in enumerate(viz_data):
            if isinstance(viz_obj, QuantumCircuit):
                # Save bloch sphere as an in-memory object.
                self.save_bloch_as_obj(quantum_circuit=viz_obj, frame=frame)

                # Insert the Bloch Sphere image data
                img_ax[viz_idx].imshow(Image.open(deepcopy(self.bloch_data)))

            if isinstance(viz_obj, ndarray):
                # Insert the calculated Julia Set data as an image
                img_ax[viz_idx].imshow(viz_obj, cmap='magma')

            # Turn off axis lines and labels
            img_ax[viz_idx].axis('off')

        # Finally show the image
        plt.show()
        plt.close()

    def qf_gif_animation(self, viz_data: Union[ndarray, QuantumCircuit], frame: int = 0) -> None:
        if self.gif_fig is None and self.gif_ax is None:
            plt.rcParams['font.family'] = prop.get_name()
            plt.rcParams["figure.figsize"] = (20, 5)
            plt.rcParams['figure.dpi'] = 60

            # Dynamically change the number of items included in the final output images
            self.gif_fig, self.gif_ax = plt.subplots(nrows=1, ncols=len(viz_data))
            self.gif_fig.subplots_adjust(top=0.85, wspace=0.1, left=0, bottom=0.1, right=1)
            self.gif_gs = self.gif_ax[0].get_gridspec()
            self.camera = Camera(self.gif_fig)
            clear_output(wait=True)

        # Ensure each item in the visualization data is added to the figure
        for gif_img_index, gif_img_obj in enumerate(viz_data):
            if isinstance(gif_img_obj, QuantumCircuit):
                # Save bloch sphere as an in-memory object and insert Bloch Sphere image data
                self.save_bloch_as_obj(quantum_circuit=gif_img_obj, frame=frame)
                self.gif_ax[gif_img_index].imshow(Image.open(deepcopy(self.bloch_data)))
            if isinstance(gif_img_obj, ndarray):
                # Insert the calculated Julia Set data as an image
                self.gif_ax[gif_img_index].imshow(gif_img_obj, cmap='magma')

        for col in range(0, self.gif_gs.ncols):
            self.gif_ax[col].axis('off')

        # Finally take a snapshot of the image
        self.camera.snap()
        plt.close()

    def save_gif_animation(self, blit:bool = True, interval_ms:int = 200, no_frames:int = 60, no_qubits: int = 1):
        anim = self.camera.animate(blit=blit, interval=interval_ms)
        anim.save(f'img/Quantum_Fractal_Animation_{no_frames}_frames_{no_qubits}_qubits.gif', writer='ffmpeg')

        clear_output(wait=True)
        with open(f"img/Quantum_Fractal_Animation_{no_frames}_frames_{no_qubits}_qubits.gif", 'rb') as fd:
            b64 = base64.b64encode(fd.read()).decode('ascii')
        return HTML(f'<img src="data:image/gif;base64,{b64}" />')

    # noinspection SpellCheckingInspection
    def qf_interactive_animation(self, quantum_circuit, frame_no, z_arr, con_arr, div_arr, height, width, interval):
        plt.rcParams['font.family'] = prop.get_name()
        plt.rcParams["animation.html"] = "jshtml"
        plt.rcParams['figure.dpi'] = 60
        plt.rcParams["figure.figsize"] = (20, 5)

        anim_fig, anim_ax = plt.subplots(nrows=1, ncols=4)
        anim_fig.subplots_adjust(top=0.85, wspace=0.1, left=0, bottom=0.1, right=1)
        anim_gs = anim_ax[0].get_gridspec()

        # Initiate the two classes responsible for generating the Bloch's sphere and visualizations
        fractal_circuit = FractalQuantumCircuit(quantum_circuit=quantum_circuit, total_number_of_frames=frame_no)

        def animate(index):
            for col in range(0, anim_gs.ncols):
                anim_ax[col].cla()

            cno, ccircuit, ccn = fractal_circuit.get_quantum_circuit(frame_iteration=index)
            anim_ax[0].imshow(Image.open(self.save_bloch_as_obj(quantum_circuit=ccircuit, return_obj=True)))
            anim_ax[1].imshow(set_1cn0(c=cno, z=z_arr.copy(), con=con_arr.copy(), div=div_arr.copy(), height=height, width=width), cmap='magma')
            anim_ax[2].imshow(set_2cn1(c=ccn, z=z_arr.copy(), con=con_arr.copy(), div=div_arr.copy(), height=height, width=width), cmap='magma')
            anim_ax[3].imshow(set_2cn2(c=ccn, z=z_arr.copy(), con=con_arr.copy(), div=div_arr.copy(), height=height, width=width), cmap='magma')

            for col in range(0, anim_gs.ncols):
                anim_ax[col].axis('off')

            # Create the title for the animation
            complex_numb = round(cno.real, 2) + round(cno.imag, 2) * 1j
            complex_amp1 = round(ccn[0].real, 2) + round(ccn[0].imag, 2) * 1j
            complex_amp2 = round(ccn[1].real, 2) + round(ccn[1].imag, 2) * 1j
            anim_fig.suptitle(f"Frame i = {index:<3} | One complex no = {complex_numb:<13} | Complex amplitude one: "
                              f"{complex_amp1:<13} and two {complex_amp2:<13}", fontsize=14)

        interactive_animation = animation.FuncAnimation(anim_fig, animate, frames=frame_no, interval=interval)
        plt.close()
        return interactive_animation
