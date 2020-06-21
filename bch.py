'''
BCH code implementation. Provides encoding/decoding methods to be used by query.py and reconstruct.py.
'''

import numpy as np
from scipy import linalg
from functools import reduce
import warnings

from galois import GaloisElement

def make_minimal_polynomial(q = 2, m = 4, d = 1):
    '''
    Returns a minimal polynomial in MATLAB order (biggest exponent first) for the element alpha in GF(q ** m).
    Currently only supports q = 2.
    Note: the minimal polynomial is NOT itself an element of the Galois field.

    '''
    # https://math.stackexchange.com/questions/2232179/how-to-find-minimal-polynomial-for-an-element-in-mboxgf2m/2869636
    idx = 0
    alpha = GaloisElement(np.array([1, 0]), q=q, m=m, poly_index=idx) ** d
    A = np.vstack([alpha ** i for i in range(m + 1)]).T
    if q != 2:
        warnings.warn("tbd how to vary parameters for q > 2")
    deg = m
    while deg > 0:
        A[0, deg] = 1
        b = np.zeros_like(A[0])
        b[0] = 1 
        if np.linalg.matrix_rank(A) == (m + 1):
            space = np.linalg.solve(A, b)
            return tuple(int(s) % q for s in space[::-1])
        else:
            A[0, deg] = 0
            deg -= 1
    raise NotImplementedError("should never hit this")

def lcm(polynomials):
    '''
    Takes the LCM of a list of minimal polynomials as returned by make_minimal_polynomial.
    '''
    return reduce(np.convolve, set(polynomials))

def generator_polynomial(q = 2, m = 4, d = 3):
    return np.mod(np.array(lcm([make_minimal_polynomial(q = q, m = m, d = i) for i in range(1, d)]), dtype=int), q)

if __name__ == "__main__":
    print(generator_polynomial(d = 4))
