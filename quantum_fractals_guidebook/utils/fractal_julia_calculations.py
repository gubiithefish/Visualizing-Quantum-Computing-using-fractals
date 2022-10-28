#!/usr/bin/env python
# coding: utf-8
# Credits: https://github.com/wmazin/Visualizing-Quantum-Computing-using-fractals

#############################################################
from numba import jit, prange
import numpy as np
import numba as nb


@jit(nopython=True, cache=False, parallel=True, error_model='numpy')
def set_1cn0(c:nb.complex128, z:np.ndarray[nb.complex128, nb.complex128], con:np.ndarray[nb.bool_, nb.bool_],
             div:np.ndarray[nb.int_, nb.int_], max_iterations: nb.int64 = 100, escape_number: nb.int64 = 2,
             height: nb.int64 = 200, width: nb.int64 = 200) -> np.ndarray[nb.int_, nb.int_]:
    for x in prange(width):
        for y in prange(height):
            for j in range(max_iterations):
                if con[x, y]:
                    z[x, y] = z[x, y] ** 2 + c
                    if abs(z[x, y]) > escape_number:
                        con[x, y] = False
                        div[x, y] = j
    return div


@jit(nopython=True, cache=False, parallel=True, error_model='numpy')
def set_2cn1(c:[nb.complex128], z:np.ndarray[nb.complex128, nb.complex128], con:np.ndarray[nb.bool_, nb.bool_],
             div:np.ndarray[nb.int_, nb.int_], max_iterations: nb.int64 = 100, escape_number: nb.int64 = 2,
             height: nb.int64 = 200, width: nb.int64 = 200) -> np.ndarray[nb.int_, nb.int_]:
    for x in prange(width):
        for y in prange(height):
            for j in range(max_iterations):
                if con[x, y]:
                    z[x, y] = (z[x, y] ** 2 + c[0]) / (z[x, y] ** 2 + c[1])
                    if abs(z[x, y]) > escape_number:
                        con[x, y] = False
                        div[x, y] = j
    return div


@jit(nopython=True, cache=False, parallel=True, error_model='numpy')
def set_2cn2(c:[nb.complex128], z:np.ndarray[nb.complex128, nb.complex128], con:np.ndarray[nb.bool_, nb.bool_],
             div:np.ndarray[nb.int_, nb.int_], max_iterations: nb.int64 = 100, escape_number: nb.int64 = 2,
             height: nb.int64 = 200, width: nb.int64 = 200) -> np.ndarray[nb.int_, nb.int_]:
    for x in prange(width):
        for y in prange(height):
            for j in range(max_iterations):
                if con[x, y]:
                    z[x, y] = (c[0] * z[x, y] ** 2 + 1 - c[0]) / (c[1] * z[x, y] ** 2 + 1 - c[1])
                    if abs(z[x, y]) > escape_number:
                        con[x, y] = False
                        div[x, y] = j
    return div


# @jit(nopython=True, cache=False, parallel=True, nogil=True)
# def julia_set_jit(statevector_data: np.ndarray, height: int = heightsize, width: int = widthsize, x: int = 0, y: int = 0, zoom: int = 1, max_iterations: int = 100):
#     # To make navigation easier we calculate these values
#     x_width: float = 1.5
#     x_from: float = x - x_width / zoom
#     x_to: float = x + x_width / zoom
#
#     y_height: float = 1.5 * height / width
#     y_from: float = y - y_height / zoom
#     y_to: float = y + y_height / zoom
#
#     # Here the actual algorithm starts and the z parameter is defined for the Julia set function
#     x = np.linspace(x_from, x_to, width).reshape((1, width))
#     y = np.linspace(y_from, y_to, height).reshape((height, 1))
#     z = (x + 1j * y)
#     # To keep track in which iteration the point diverged
#     div_time = np.full(z.shape, max_iterations - 1, dtype=numba.uint8)
#
#     # To keep track on which points did not converge so far
#     m = np.full(z.shape, True, dtype=numba.bool_)
#
#     for x in prange(size):
#         for y in prange(size):
#             for j in range(max_iterations):
#                 if m[x, y]:
#                     # Create first sub-equation of Julia mating equation
#                     uval = z[x, y] ** (upper_pwrs[0])
#                     lval = z[x, y] ** (lower_pwrs[0])
#
#                     # Add middle sub-equation(s) of Julia mating equation
#                     for i in prange(n_sub_expressions - 1):
#                         uval = uval + statevector_data[upper_idxs[i]] * z[x, y] ** (upper_pwrs[i + 1])
#                         lval = lval + statevector_data[lower_idxs[i]] * z[x, y] ** (lower_pwrs[i + 1])
#
#                     # Add final sub-equation of Julia mating equation
#                     uval = uval + statevector_data[upper_idxs[n_sub_expressions - 1]]
#                     lval = lval + statevector_data[lower_idxs[n_sub_expressions - 1]]
#
#                     z[x, y] = uval / lval
#                     if abs(z[x, y]) > escapeno:
#                         m[x, y] = False
#                         div_time[x, y] = j
#     return div_time