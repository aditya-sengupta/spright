'''
BCH code implementation. Provides encoding/decoding methods to be used by query.py and reconstruct.py.
'''

from galois import GaloisElement

def make_generating_polynomial(q, m, d):
    alpha = np.array([1, 0]) # use alpha = x as the primitive element: can change later
    