import numpy as np

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

def binary_ints(n):
    '''
    Returns a matrix where row 'i' is dec_to_bin(i, n), for i from 0 to 2 ** n - 1.
    From https://stackoverflow.com/questions/28111051/create-a-matrix-of-binary-representation-of-numbers-in-python.
    '''
    a = np.arange(2 ** n, dtype=int)[np.newaxis,:]
    b = np.arange(n, dtype=int)[::-1,np.newaxis]
    return np.array(a & 2**b > 0, dtype=int)
