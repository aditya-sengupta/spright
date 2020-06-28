'''
Methods for the query generator: specifically, to

1. generate sparsity coefficients b and subsampling matrices M
2. get the indices of a signal subsample
3. compute a subsampled and delayed Walsh-Hadamard transform.
'''

import numpy as np

from utils import fwht, bin_to_dec, dec_to_bin, binary_ints

def get_b_simple(signal):
    '''
    A semi-arbitrary fixed choice of the sparsity coefficient. See get_b for full signature.
    '''
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
    '''
    A semi-arbitrary fixed choice of the subsampling matrices. See get_Ms for full signature.
    '''
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

def get_Ms_BCH(n, b, num_to_get=None):
    '''
    The subsampling matrices that enable BCH coding. See get_Ms for full signature.
    '''
    if n % b != 0:
        raise NotImplementedError("b must be exactly divisible by n")
    if num_to_get is None:
        num_to_get = n // b
    return

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

def get_D_identity_like(n, **kwargs):
    '''
    Gets the delays matrix [0; I], of dimension (n+1, n). See get_D for full signature.
    # TODO: rename this to avoid conceptual name collision with zeros_like or empty_like
    '''
    return np.vstack((np.zeros(n,), np.eye(n)))

def get_D_random(n, **kwargs):
    '''
    Gets a random delays matrix of dimension (num_delays, n). See get_D for full signature.
    '''
    num_delays = kwargs.get("num_delays")
    choices = np.random.choice(2 ** n, num_delays, replace=False)
    return np.array([dec_to_bin(x, n) for x in choices])

def get_D_nso(n, **kwargs):
    '''
    Get a repetition code based (NSO-SPRIGHT) delays matrix. See get_D for full signature.
    '''
    num_delays = kwargs.get("num_delays")
    p1 = num_delays // n # is this what we want?
    random_offsets = get_D_random(n, num_delays=p1) # OR np.random.binomial(1, 0.5, (num_delays, p1))
    D = np.empty((0, n))
    identity_like = get_D_identity_like(n)
    for row in random_offsets:
        modulated_offsets = (row + identity_like) % 2
        D = np.vstack((D, modulated_offsets))
    return D
    
def get_D(n, method="random", **kwargs):
    '''
    Delay generator: gets a delays matrix.

    Arguments
    ---------
    n : int
    number of bits: log2 of the signal length.

    Returns
    -------
    D : numpy.ndarray of binary ints, dimension (num_delays, n).
    The delays matrix; if num_delays is not specified in kwargs, see the relevant sub-function for a default.
    '''
    return {
        "identity_like" : get_D_identity_like,
        "random" : get_D_random
    }.get(method)(n, **kwargs)

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

def compute_delayed_wht(signal, M, D):
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
    samples_to_transform = signal.signal_t[np.array([subsample_indices(M, d) for d in D])] # subsample to allow small WHTs
    return np.array([fwht(row) for row in samples_to_transform]) # compute the small WHTs
    