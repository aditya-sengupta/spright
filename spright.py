'''
SPRIGHT decoding main file. Logic flow:

1. Generate a signal from signal.py
2. Subsample from query.py
3. Peel using reconstruct.py
'''

import numpy as np
from matplotlib import pyplot as plt

from utils import fwht, dec_to_bin, bin_to_dec, binary_ints
from query import compute_delayed_wht, get_Ms, get_b
from reconstruct import singleton_detection

class SPRIGHT:
    '''
    Class to store encoder/decoder configurations and carry out encoding/decoding.

    Attributes
    ----------
    query_method : str
    The method to generate the sparsity coefficient and the subsampling matrices.
    Currently implemented methods: 
        "simple" : choose some predetermined matrices based on problem size.

    reconstruct_method : str
    The method to detect singletons.
    Currently implemented methods:
        "noiseless" : decode according to [2], section 4.2, with the assumption the signal is noiseless
        "mle" : naive noisy decoding; decode by taking the maximum-likelihood singleton that could be at that bin.
    '''
    def __init__(self, query_method, reconstruct_method):
        self.query_method = query_method
        self.reconstruct_method = reconstruct_method

    def transform(self, signal, verbose=False):
        '''
        Full SPRIGHT encoding and decoding. Implements Algorithms 1 and 2 from [2].
        (numbers) in the comments indicate equation numbers in [2].
        
        Arguments
        ---------
        signal : Signal object.
        The signal to be transformed / compared to.
        
        verbose : boolean
        Whether to print intermediate steps.

        Returns
        -------
        wht : ndarray
        The WHT constructed by subsampling and peeling.
        '''

        result = []
        wht = np.zeros_like(signal.signal_t)
        b = get_b(signal, method=self.query_method)
        Ms = get_Ms(signal.n, b, method=self.query_method)
        Us, Ss = [], []
        singletons = {}
        multitons = []
        num_delays = signal.n + 1
        K = binary_ints(signal.n)
            
        # subsample, make the observation [U] and delays [D] matrices
        for M in Ms:
            U, D = compute_delayed_wht(signal, M, num_delays, force_identity_like=(self.reconstruct_method == "noiseless"))
            Us.append(U)
            Ss.append((-1) ** (D @ K)) # offset signature matrix
        
        cutoff = 2 * signal.noise_sd ** 2 * (2 ** (signal.n - b)) * num_delays # noise threshold
        if verbose:
            print('cutoff: {}'.format(cutoff))
        # K is the binary representation of all integers from 0 to 2 ** n - 1.
        select_froms = np.array([[bin_to_dec(row) for row in M.T.dot(K).T] for M in Ms])
        # `select_froms` is the collection of 'j' values and associated indices
        # so that we can quickly choose from the coefficient locations such that M.T @ k = j as in (20)
        # example: ball j goes to bin at "select_froms[i][j]"" in stage i
        
        # begin peeling
        # index convention for peeling: 'i' goes over all M/U/S values
        # i.e. it refers to the index of the subsampling group (zero-indexed - off by one from the paper).
        # 'j' goes over all columns of the WHT subsample matrix, going from 0 to 2 ** b - 1.
        # e.g. (i, j) = (0, 2) refers to subsampling group 0, and aliased bin 2 (10 in binary)
        # which in the example of section 3.2 is the multiton X[0110] + X[1010] + W1[10]
        
        # a multiton will just store the (i, j)s in a list
        # a singleton will map from the (i, j)s to the true (binary) values k.
        # e.g. the singleton (0, 0), which in the example of section 3.2 is X[0100] + W1[00]
        # would be stored as the dictionary entry (0, 0): array([0, 1, 0, 0]).
        
        there_were_multitons = True
        while there_were_multitons:
            if verbose:
                print('-----')
                print('the measurement matrix')
                for U in Us:
                    print(U)
            
            # first step: find all the singletons and multitons.
            singletons = {} # dictionary from (i, j) values to the true index of the singleton, k.
            multitons = [] # list of (i, j) values indicating where multitons are.
            
            for i, (U, S, select_from) in enumerate(zip(Us, Ss, select_froms)):
                for j, col in enumerate(U.T):
                    # note that np.inner(x, x) is used as norm-squared: marginally faster than taking norm and squaring
                    if np.inner(col, col) > cutoff:
                        selection = np.where(select_from == j)[0] # pick all the k such that M.T @ k = j
                        k, sgn = singleton_detection(
                            col, 
                            method=self.reconstruct_method, 
                            selection=selection, 
                            S_slice=S[:, selection], 
                            n=signal.n
                        ) # find the best fit singleton
                        k_dec = bin_to_dec(k)
                        rho = np.dot(S[:,k_dec], col)*sgn/len(col)                    
                        residual = col - sgn * rho * S[:,k_dec] 
                        if verbose:
                            print((i, j), np.inner(residual, residual))
                        if np.inner(residual, residual) > cutoff:
                            multitons.append((i, j))
                        else: # declare as singleton
                            singletons[(i, j)] = (k, rho, sgn)
                            if verbose:
                                print('amplitude: {}'.format(rho))
                            
            
            # all singletons and multitons are discovered
            if verbose:
                print('singletons:')
                for ston in singletons.items():
                    print("\t{0} {1}\n".format(ston, bin_to_dec(ston[1][0])))

                print("Multitons : {0}\n".format(multitons))
            
            # WARNING: this is not a correct thing to do
            # in the last iteration of peeling, everything will be singletons and there
            # will be no multitons
            if len(multitons) == 0: # no more multitons, and can construct final WHT
                there_were_multitons = False
                
            # balls to peel
            balls_to_peel = set()
            ball_values = {}
            ball_sgn = {}
            for (i, j) in singletons:
                k, rho, sgn = singletons[(i, j)]
                ball = bin_to_dec(k)
                balls_to_peel.add(ball)
                
                ball_values[ball] = rho
                ball_sgn[ball] = sgn
                
            if verbose:
                print('these balls will be peeled')
                print(balls_to_peel)
            # peel
            for ball in balls_to_peel:
                k = dec_to_bin(ball, signal.n)
                potential_peels = [(l, bin_to_dec(M.T.dot(k))) for l, M in enumerate(Ms)]

                result.append((k, ball_sgn[ball]*ball_values[ball]))
                
                for peel in potential_peels:
                    signature_in_stage = Ss[peel[0]][:,ball]
                    to_subtract = ball_sgn[ball] * ball_values[ball] * signature_in_stage
                    Us[peel[0]][:,peel[1]] -= to_subtract
                    if verbose:
                        print('this is subtracted:')
                        print(to_subtract)
                        print("Peeled ball {0} off bin {1}".format(bin_to_dec(k), peel))
            
        for k, value in result: # iterating over (i, j)s
            idx = bin_to_dec(k) # converting 'k's of singletons to decimals
            if wht[idx] == 0:
                wht[idx] = value
            else:
                wht[idx] = (wht[idx] + value) / 2 
                # average out noise; e.g. in the example in 3.2, U1[11] and U2[11] are the same singleton,
                # so averaging them reduces the effect of noise.
        
        wht /= 2 ** (signal.n - b)
        return wht

if __name__ == "__main__":
    from inputsignal import Signal
    test_signal = Signal(8, [4, 6, 10, 15], strengths=[2, 4, 1, 1], noise_sd=0.01)
    spright = SPRIGHT(query_method="bch", reconstruct_method="mle")
    residual = spright.transform(test_signal) - test_signal.signal_w
    print("Residual energy: {0}".format(np.inner(residual, residual)))
