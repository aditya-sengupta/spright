'''
Methods for the reconstruction engine; specifically, to

1. carry out singleton detection
2. get the cardinalities of all bins in a subsampling group (debugging only).
'''

import numpy as np
from .utils import bin_to_dec, dec_to_bin, binary_ints, sign, flip
from .query import compute_delayed_wht

def singleton_detection_noiseless(U_slice, **kwargs):
    '''
    Finds the true index of a singleton, or the best-approximation singleton of a multiton.
    Assumes P = n + 1 and D = [0; I].
    
    Arguments
    ---------
    U_slice : numpy.ndarray, (P,).
    The WHT component of a subsampled bin, with element i corresponding to delay i.
    
    Returns
    -------
    k : numpy.ndarray
    Index of the corresponding right node, in binary form.
    '''
    return (-np.sign(U_slice * U_slice[0])[1:] == 1).astype(np.int), 1

def singleton_detection_mle(U_slice, **kwargs):
    '''
    Finds the true index of a singleton, or the best-approximation singleton of a multiton, in the presence of noise.
    Uses MLE: looks at the residuals created by peeling off each possible singleton.
    
    Arguments
    ---------
    U_slice : numpy.ndarray, (P,).
    The WHT component of a subsampled bin, with element i corresponding to delay i.

    selection : numpy.ndarray.
    The decimal preimage of the bin index, i.e. the list of potential singletons whose signature under M could be the j of the bin.

    S_slice : numpy.ndarray
    The set of signatures under the delays matrix D associated with each of the elements of 'selection'.

    n : int
    The signal's number of bits.

    Returns
    -------
    k : numpy.ndarray, (n,)
    The index of the singleton.

    sgn : -1 or 1
    The sign of the bin: whether to add or subtract the WHT of this singleton from a multiton.
    '''
    selection, S_slice, n = kwargs.get("selection"), kwargs.get("S_slice"), kwargs.get("n")
    P = S_slice.shape[0]
    alphas = (1/P) * np.dot(S_slice.T, U_slice)
    residuals = np.linalg.norm(U_slice - (alphas * S_slice).T, ord=2, axis=1)
    k_sel = np.argmin(residuals)
    return dec_to_bin(selection[k_sel], n), np.sign(alphas[k_sel])

def singleton_detection_nso(U_slice, **kwargs):
    n = kwargs.get("n")
    chunks = sign(np.reshape(U_slice, (len(U_slice) // (n + 1), n + 1)))
    chunks = (np.mod((chunks.T + chunks[:,0]).T, 2)).astype(dtype=int)[:,1:]
    choices = np.vstack((np.sum(chunks, axis=0), np.sum([flip(c) for c in chunks], axis=0)))
    nso_k = np.argmin(choices, axis=0)
    return nso_k, 1

def singleton_detection(U_slice, method="mle", **kwargs):
    return {
        "mle" : singleton_detection_mle,
        "noiseless" : singleton_detection_noiseless,
        "nso" : singleton_detection_nso
    }.get(method)(U_slice, **kwargs)

def bin_cardinality(signal, M, D):
    '''
    Computes delayed WHT observations and declares cardinality based on that.
    2 is a stand-in for any cardinality > 1. For debugging purposes.
    
    Arguments
    ---------
    signal : InputSignal
    The input signal object.

    M : numpy.ndarray, shape (n, b)
    The subsampling matrix; takes on binary values.
    
    D : numpy.ndarray of ints, shape (num_delays, n).
    The delays matrix.

    Returns
    -------
    cardinality : numpy.ndarray
    0 or 1 if the bin is a zeroton or singleton resp.; 2 if multiton.
    
    singleton_indices : list
    A list (in decimal form for compactness) of the k values of the singletons. 
    Length matches the number of 1s in cardinality.
    '''
    b = M.shape[1]
    U = compute_delayed_wht(signal, M, D)
    cardinality = np.ones((signal.n,), dtype=np.int) # vector of indicators
    singleton_indices = []
    cutoff = 2 * signal.noise_sd ** 2 * (2 ** (signal.n - b)) * D.shape[0]
    if signal.noise_sd > 0:
        K = binary_ints(signal.n)
        S = (-1) ** (D @ K)
    for i, col in enumerate(U.T):
        sgn = 1
        print("Column:   ", col)
        # <col, col> = |col|^2 = |U|^2
        if np.inner(col, col) <= cutoff:
            cardinality[i] = 0
        else:
            if signal.noise_sd == 0:
                k = singleton_detection_noiseless(col)
            else:
                selection = np.where([bin_to_dec(row) == i for row in (M.T.dot(K)).T])[0]
                k, sgn = singleton_detection(col, method="mle", selection=selection, S_slice=S[:, selection], n=signal.n)
            rho = np.mean(np.abs(col))
            residual = col - sgn * rho * (-1) ** np.dot(D, k)
            print("Residual: ", residual)
            if np.inner(residual, residual) > cutoff:
                cardinality[i] = 2
            else:
                singleton_indices.append(bin_to_dec(k))
    return cardinality, singleton_indices
    