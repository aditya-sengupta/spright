'''
Methods for the query generator: specifically, to

1. generate sparsity coefficients b and subsampling matrices M
2. get the indices of a signal subsample
3. compute a subsampled and delayed Walsh-Hadamard transform.
'''

import numpy as np

from utils import fwht, bin_to_dec, dec_to_bin, binary_ints

def get_b_simple(signal):
    return signal.n // 2

def get_b(signal, method="simple"):
    '''
    Get the sparsity coefficient for the signal.

    Arguments
    ---------
    signal : Signal
    The signal whose WHT we want.

    method : str
    The method to use. All methods referenced must use this signature (minus "method".)

    Returns
    -------
    b : int
    The sparsity coefficient.
    '''
    return {
        "simple" : get_b_simple
    }.get(method)(signal)


def get_Ms_simple(n, b, num_to_get=None):
    if n % b != 0:
        raise NotImplementedError("b must be exactly divisible by n")
    if num_to_get is None:
        num_to_get = n // b
    
    Ms = []
    for i in range(num_to_get - 1, -1, -1):
        M = np.zeros((n, b))
        M[(b * i) : (b * (i + 1))] = np.eye(b)
        Ms.append(M)

    return Ms

def get_Ms(n, b, num_to_get=None, method="simple"):
    '''
    Gets subsampling matrices for different sparsity levels.

    Arguments
    ---------
    n : int
    log2 of the signal length.

    b : int
    Sparsity.

    num_to_get : int
    The number of M matrices to return.

    method : str
    The method to use. All methods referenced must use this signature (minus "method".)

    Returns
    -------
    Ms : list of numpy.ndarrays, shape (n, b)
    The list of subsampling matrices.
    '''
    return {
        "simple" : get_Ms_simple
    }.get(method)(n, b, num_to_get)

def subsample_indices(M, d):
    '''
    Query generator: creates indices for signal subsamples.
    
    Arguments
    ---------    
    M : numpy.ndarray, shape (n, b)
    The subsampling matrix; takes on binary values.
    
    d : numpy.ndarray, shape (n,)
    The subsampling offset; takes on binary values.
    
    Returns
    -------
    indices : numpy.ndarray, shape (B,)
    The (decimal) subsample indices. Mostly for debugging purposes.
    '''
    L = binary_ints(M.shape[1])
    inds_binary = np.mod(np.dot(M, L).T + d, 2).T 
    return bin_to_dec(inds_binary)

def compute_delayed_wht(signal, M, num_delays, force_identity_like=False):
    '''
    Creates random delays, subsamples according to M and the random delays,
    and returns the subsample WHT along with the delays.

    Arguments
    ---------
    signal : Signal object
    The signal to subsample, delay, and compute the WHT of.

    M : numpy.ndarray, shape (n, b)
    The subsampling matrix; takes on binary values.

    num_delays : int
    The number of delays to apply; or, the number of rows in the delays matrix.

    force_identity_like : boolean
    Whether to make D = [0; I] like in the noiseless case; for debugging.
    '''
    if num_delays is None:
        num_delays = signal.n + 1
    if signal.noise_sd > 0:
        if not force_identity_like:
            choices = np.random.choice(2 ** signal.n, num_delays, replace=False)
        else:
            choices = np.array([0] + [2 ** i for i in range(signal.n)])
        D = np.array([dec_to_bin(x, signal.n) for x in choices])
    else:
        D = np.vstack((np.zeros(signal.n,), np.eye(signal.n)))
    samples_to_transform = signal.signal_t[np.array([subsample_indices(M, d) for d in D])] # subsample to allow small WHTs
    U = np.array([fwht(row) for row in samples_to_transform]) # compute the small WHTs
    return U, D
