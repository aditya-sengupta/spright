'''
Computations over finite fields. Exists in external libraries, but none that are canonical and up-to-date,
so easier just to rewrite.
'''
import numpy as np
from functools import reduce
from utils import polymod

# TODO: fill this in with a function rather than manually
prime_polynomials = {
    8 : [[1, 0, 1, 1], [1, 1, 0, 1]],
    16: [[1, 0, 0, 1, 1]]
}
    
class GaloisElement(np.ndarray):
    '''
    An element of GF(q ** m), which we identify as the field of polynomials with coefficients in Z_q modulo 
    a prime polynomial, specifically prime_polynomials.get(q ** m)[poly_index].
    '''
    def __new__(cls, poly, q=2, m=1, poly_index=0, qpoly=None):
        if qpoly is None:
            qpoly = np.array(prime_polynomials.get(q ** m)[poly_index])
        polynomial = np.asarray(polymod(poly, qpoly, q), dtype=int).view(cls)
        polynomial.q = q
        polynomial.m = m
        polynomial.qpoly = qpoly
        assert all([p < q for p in polynomial]) and all([p >= 0 for p in polynomial]), "coefficients must be in Z_q"
        return polynomial

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.q = getattr(obj, 'q', None)
        self.m = getattr(obj, 'm', None)
        self.qpoly = getattr(obj, 'qpoly', None)

    def equiv(self, other):
        return all(self.qpoly == other.qpoly)

    def __repr__(self):
        return ''.join([" + " * (i > 0) + str(p) + "x^" + str(len(self) - 1 - i) for i, p in enumerate(self)]) + " in GF({0}^{1})".format(self.q, self.m)

    def pairwise_operate(self, other, opname):
        assert self.equiv(other), "polynomials are from different fields"
        operation = {"add" : np.polyadd, "mul" : np.polymul}.get(opname)
        return GaloisElement(operation(self, other), q=self.q, m=self.m, qpoly=self.qpoly)
        
    def __add__(self, other):
        if not isinstance(other, GaloisElement):
            return super().__add__(other)
        return self.pairwise_operate(other, "add")

    def __radd__(self, other):
        if isinstance(other, int) and other == 0: # edge case for sum
            return self
        return other.__add__(self)

    def __mul__(self, other):
        if not isinstance(other, GaloisElement):
            return super().__mul__(other)
        return self.pairwise_operate(other, "mul")

    def __pow__(self, n):
        if n == 0:
            return GaloisElement([1], q=self.q, m=self.m, qpoly=self.qpoly)
        return self * self ** (n - 1)

    def compose(self, other):
        '''
        Composes self onto other, in the order self(other(x)).
        Alternatively: substitutes other into self.
        '''
        print([coeff * other ** (len(self) - 1 - i) for i, coeff in enumerate(self)])
        return np.mod(sum([coeff * other ** i for i, coeff in enumerate(self)]), self.q)

if __name__ == "__main__":
    from copy import deepcopy
    a = GaloisElement([1, 0, 1], q=2, m=4)
    b = GaloisElement([1, 1], q=2, m=4)
    print(a.compose(b))
