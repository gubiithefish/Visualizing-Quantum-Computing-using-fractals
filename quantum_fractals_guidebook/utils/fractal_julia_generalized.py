#!/usr/bin/env python
# coding: utf-8
# Credits & License: https://github.com/wmazin/Visualizing-Quantum-Computing-using-fractals

#############################################################
from numpy import uint8, uint16, uint32, int32, int64, linspace, bool_, complex_, ndarray, array
from numba import jit, prange


@jit(nopython=True, cache=False, parallel=True, nogil=True, error_model='numpy')
def set_general(c: ndarray[complex_], z: ndarray[complex_, complex_],
                con: ndarray[bool_, bool_], div: ndarray[uint16, uint16],
                upper_pwrs: ndarray[uint32] = array([1]), upper_idxs: ndarray[uint32] = array([0]),
                lower_pwrs: ndarray[uint32] = array([1]), lower_idxs: ndarray[uint32] = array([1]),
                max_iterations: uint16 = 100, number_of_qubits: uint8 = 1, escape_number: uint8 = 2,
                height: uint16 = 200, width: uint16 = 200,) -> ndarray[uint16, uint16]:
    """
    n-qubit mating:
        z^2^(n-1) + c[2^n-2] * z^(2^(n-1)-1) + c[2^n-4] * z^(2^(n-1)-2) + ... * c[6] * z^2 + c[4] + c[2] * z + c[0]
    z = ───────────────────────────────────────────────────────────────────────────────────────────────────────────
        z^2^(n-1) + c[2^n-1] * z^(2^(n-1)-1) + c[2^n-3] * z^(2^(n-1)-2) + ... * c[7] * z^2 + c[5] + c[3] * z + c[1]

    where c is the statevector for the n-th qubit

    example for 3-qubit Julia Set Mating:
        z^4 + c[6] * z^3 + c[4] + z^2 * c[2] * z + c[0]
    z = ───────────────────────────────────────────────
        z^4 + c[7] * z^3 + c[5] + z^2 * c[3] * z + c[1]
    """
    for x in prange(width):
        for y in prange(height):
            for j in range(max_iterations):
                if con[x, y]:
                    # Create first sub-equation of Julia mating equation
                    upper_val = z[x, y] ** (upper_pwrs[0])
                    lower_val = z[x, y] ** (lower_pwrs[0])

                    # Add middle sub-equation(s) of Julia mating equation
                    for i in prange(((2 ** number_of_qubits) // 2) - 1):
                        upper_val = upper_val + c[upper_idxs[i]] * z[x, y] ** (upper_pwrs[i + 1])
                        lower_val = lower_val + c[lower_idxs[i]] * z[x, y] ** (lower_pwrs[i + 1])

                    # Add final sub-equation of Julia mating equation
                    upper_val = upper_val + c[upper_idxs[((2 ** number_of_qubits) // 2) - 1]]
                    lower_val = lower_val + c[lower_idxs[((2 ** number_of_qubits) // 2) - 1]]

                    z[x, y] = upper_val / lower_val
                    if abs(z[x, y]) > escape_number:
                        con[x, y] = False
                        div[x, y] = j
    return div


def get_fraction_powers_and_indices(no_qubits: int64 = 1, power_offset: int64 = 0):
    """Returns the powers and indices for any given amount of Qubits used in the <set_general> function"""
    # Adjust the start and stop integer values for the powers
    powers_start = 2 ** no_qubits // 2 + power_offset
    powers_stop = 1 + power_offset

    # Calculate all the evenly spaced specified intervals
    upper_pwrs = linspace(start=powers_start, stop=powers_stop, num=2 ** no_qubits // 2).astype(int32)
    upper_idxs = linspace(start=2 ** no_qubits - 2, stop=0, num=2 ** no_qubits // 2).astype(int32)
    lower_pwrs = linspace(start=powers_start, stop=powers_stop, num=2 ** no_qubits // 2).astype(int32)
    lower_idxs = linspace(start=2 ** no_qubits - 1,  stop=1, num=2 ** no_qubits // 2).astype(int32)

    # Return
    return upper_pwrs, upper_idxs, lower_pwrs, lower_idxs


if __name__ == "__main__":
    def pretty_print(upper_pwrs, upper_idxs, lower_pwrs, lower_idxs):
        upper = [f"z^{upper_pwrs[i]} + out_data[{upper_idxs[i]}]" for i in range(len(upper_pwrs))]
        lower = [f"z^{lower_pwrs[i]} + out_data[{lower_idxs[i]}]" for i in range(len(lower_pwrs))]
        print(" " * 4 + " * ".join(upper) + f"\nz = {len(' * '.join(upper)) * '─'}\n" + " " * 4 + " * ".join(lower))

    pretty_print(*get_fraction_powers_and_indices(no_qubits=1, power_offset=0)); print("")
    pretty_print(*get_fraction_powers_and_indices(no_qubits=2, power_offset=0)); print("")
    pretty_print(*get_fraction_powers_and_indices(no_qubits=3, power_offset=0)); print("")
    pretty_print(*get_fraction_powers_and_indices(no_qubits=4, power_offset=0)); print("")