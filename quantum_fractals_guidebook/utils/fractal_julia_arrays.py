

# Importing standard python libraries
from typing import Union

# Import externally installed libraries
import numpy as np


class GetJuliaArrays:
    def __init__(self, julia_iterations: int = 100, x_start: Union[int, float] = 0.0, x_width: Union[int, float] = 1.5,
                 y_start: Union[int, float] = 0.0, y_width: Union[int, float] = 1.5, height: Union[int, float] = 500,
                 width: Union[int, float] = 500, zoom: Union[int, float] = 1.0) -> None:
        self.julia_iterations = julia_iterations
        self.x_start = x_start
        self.x_width = x_width
        self.y_start = y_start
        self.y_width = y_width
        self.height = height
        self.width = width
        self.zoom = zoom

    def get_z_array(self) -> np.ndarray:
        x_min: float = self.x_start - self.x_width / self.zoom
        x_max: float = self.x_start + self.x_width / self.zoom
        y_min: float = self.y_start - self.y_width / self.zoom
        y_max: float = self.y_start + self.y_width / self.zoom

        x_arr = np.linspace(start=x_min, stop=x_max, num=self.width).reshape((1, self.width))
        y_arr = np.linspace(start=y_min, stop=y_max, num=self.height).reshape((self.height, 1))
        return x_arr + 1j * y_arr

    def get_diverged_array(self) -> np.ndarray:
        """To keep track in which iteration the point diverged"""
        return np.full(self.get_z_array().shape, self.julia_iterations - 1, dtype='int64')

    def get_converging_array(self) -> np.ndarray:
        """To keep track on which points did not converge so far"""
        return np.full(self.get_z_array().shape, True, dtype=np.bool_)