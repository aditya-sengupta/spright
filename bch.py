'''
BCH code implementation. Provides encoding/decoding methods to be used by query.py and reconstruct.py.
'''

import numpy as np
from scipy import linalg
from galois import GaloisElement
import warnings

def make_minimal_polynomial(q=2, m=4, d=1):
    '''
    Returns a minimal polynomial in MATLAB order (biggest exponent first) for the element alpha in GF(q ** m).
    Currently only supports q = 2.
    Note: the minimal polynomial is NOT itself an element of the Galois field.

    '''
    # https://math.stackexchange.com/questions/2232179/how-to-find-minimal-polynomial-for-an-element-in-mboxgf2m/2869636
    idx = 0
    alpha = GaloisElement(np.array([1, 0]), q=q, m=m, poly_index=idx) ** 8
    A = np.vstack([alpha ** i for i in range(m + 1)]).T
    if q != 2:
        warnings.warn("tbd how to vary parameters for q > 2")
    deg = m
    P, L, U = linalg.lu(A)
    print(P)
    print(L)
    print(U)
    while deg > 0:
        if deg < m:
            A[0, deg + 1] = 0
        A[0, deg] = 1
        b = np.zeros_like(A[:,0])
        b[0] = 1
        if np.linalg.matrix_rank(A) < m:
            deg -= 1
        else:
            space = np.linalg.solve(A, b)
            return np.asarray(space, dtype=int)[::-1] % q
    return "should never hit this"

if __name__ == "__main__":
    print(make_minimal_polynomial(m=4))
