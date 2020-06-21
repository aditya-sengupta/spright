'''
BCH code implementation. Provides encoding/decoding methods to be used by query.py and reconstruct.py.
'''

import numpy as np
from scipy import linalg
from galois import GaloisElement

def make_minimal_polynomial(q=2, m=4, d=1):
    '''
    Returns a minimal polynomial (biggest exponent first) for the element alpha in GF(q ** m). Currently only supports q = 2.
    Note: the minimal polynomial is NOT itself an element of the Galois field (I think?)

    '''
    # https://math.stackexchange.com/questions/2232179/how-to-find-minimal-polynomial-for-an-element-in-mboxgf2m/2869636
    idx = 0
    alpha = GaloisElement(np.array([1, 0]), q=q, m=m, poly_index=idx) ** 3
    system = np.vstack([alpha ** i for i in range(m + 1)]).T
    if q != 2:
        raise NotImplementedError("tbd how to vary parameters")
    '''space = linalg.null_space(system.T)
    print(space[np.argmax(space[:,-1])])
    solution = space[np.argmax(space[:,-1])] # slight stopgap'''
    # system[0, m] = 1
    print(system)
    b = np.zeros_like(system[:,0])
    b[-1] = 1
    space = np.linalg.solve(system, b)
    print(space)
    '''if space.shape[1] > 1:
        solution = space[:,np.argmax(abs(space[-1]))]
    else:
        solution = space
    return np.asarray(~np.isclose(solution, 0), dtype=np.int8).ravel()[::-1]'''

if __name__ == "__main__":
    print(make_minimal_polynomial(m=5))
