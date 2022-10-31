#!/usr/bin/env python
# coding: utf-8
# Credits & License: https://github.com/wmazin/Visualizing-Quantum-Computing-using-fractals

#############################################################
from numpy import uint8, uint16, bool_, complex_, ndarray
from numba import jit, prange


@jit(nopython=True, cache=False, parallel=True, error_model='numpy')
def set_1cn0(c: complex_, z: ndarray[complex_, complex_], con: ndarray[bool_, bool_],
             div: ndarray[uint16, uint16], max_iterations: uint16 = 100, escape_number: uint8 = 2,
             height: uint16 = 200, width: uint16 = 200,) -> ndarray[uint16, uint16]:
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
def set_2cn1(c: ndarray[complex_], z: ndarray[complex_, complex_], con: ndarray[bool_, bool_],
             div: ndarray[uint16, uint16], max_iterations: uint16 = 100, escape_number: uint8 = 2,
             height: uint16 = 200, width: uint16 = 200,) -> ndarray[uint16, uint16]:
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
def set_2cn2(c: ndarray[complex_], z: ndarray[complex_, complex_], con: ndarray[bool_, bool_],
             div: ndarray[uint16, uint16], max_iterations: uint16 = 100, escape_number: uint8 = 2,
             height: uint16 = 200, width: uint16 = 200,) -> ndarray[uint16, uint16]:
    for x in prange(width):
        for y in prange(height):
            for j in range(max_iterations):
                if con[x, y]:
                    z[x, y] = (c[0] * z[x, y] ** 2 + 1 - c[0]) / (c[1] * z[x, y] ** 2 + 1 - c[1])
                    if abs(z[x, y]) > escape_number:
                        con[x, y] = False
                        div[x, y] = j
    return div
