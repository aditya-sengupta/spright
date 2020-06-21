'''
Utility functions.
'''

import numpy as np
from functools import partial

def fwht(x):
    """Recursive implementation of the 1D Cooley-Tukey FFT"""
    # x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N == 1:
        return x
    else:
        X_even = fwht(x[0:(N//2)])
        X_odd = fwht(x[(N//2):])
        return np.concatenate([(X_even + X_odd),
                               (X_even - X_odd)])
    
def bin_to_dec(x):
    n = len(x)
    c = 2**(np.arange(n)[::-1])
    return c.dot(x).astype(np.int)

def dec_to_bin(x, num_bits):
    assert x < 2**num_bits, "number of bits are not enough"
    u = bin(x)[2:].zfill(num_bits)
    u = list(u)
    u = [int(i) for i in u]
    return np.array(u)

def binary_ints(m):
    '''
    Returns a matrix where row 'i' is dec_to_bin(i, m), for i from 0 to 2 ** m - 1.
    From https://stackoverflow.com/questions/28111051/create-a-matrix-of-binary-representation-of-numbers-in-python.
    '''
    a = np.arange(2 ** m, dtype=int)[np.newaxis,:]
    b = np.arange(m, dtype=int)[::-1,np.newaxis]
    return np.array(a & 2**b > 0, dtype=int)

def base_ints(q, m):
    '''
    Returns a matrix where row 'i' is the base-q representation of i, for i from 0 to q ** m - 1.
    Covers the functionality of binary_ints when n = 2, but binary_ints is faster for that case.
    '''
    get_row = lambda i: np.array([int(j) for j in np.base_repr(i, base=q).zfill(m)])
    return np.vstack((get_row(i) for i in range(q ** m)))

def polymod(p1, p2, q, m):
    '''
    Computes p1 modulo p2, and takes the coefficients modulo q.
    '''
    p1 = np.trim_zeros(p1, trim='f')
    p2 = np.trim_zeros(p2, trim='f')
    while len(p1) >= len(p2) and len(p1) > 0:
        p1 -= p1[0] // p2[0] * np.pad(p2, (0, len(p1) - len(p2)))
        p1 = np.trim_zeros(p1, trim='f')
    return np.pad(np.mod(p1, q), (m + 1 - len(p1), 0))

def rref(A, b, q):
    '''
    Row reduction, to easily solve finite field systems.
    '''
    raise NotImplementedError()

def polyeval(poly, x):
    return sum([coeff * x ** (len(poly) - 1 - i) for i, coeff in enumerate(poly)])
