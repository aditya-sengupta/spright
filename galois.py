'''
Computations over finite fields. Exists in external libraries, but none that are canonical and up-to-date,
so easier just to rewrite.
'''
import numpy as np
from sympy import totient


# TODO: fill this in with a function rather than manually
prime_polynomials = {
    8 : [[1, 0, 1, 1], [1, 1, 0, 1]],
    16: [[1, 0, 0, 1, 1]]
}

def polymod(p1, p2, q):
    '''
    Computes p1 modulo p2, and takes the coefficients modulo q.
    '''
    p1 = np.trim_zeros(p1, trim='f')
    while len(p1) >= len(p2) and len(p1) > 0:
        p1 -= p1[0] / p2[0] * np.pad(p2, (0, len(p1) - len(p2)))
        p1 = np.trim_zeros(p1, trim='f')
    return np.mod(p1, q)
    

class GaloisElement(np.ndarray):
    '''
    An element of GF(q^m), which we identify as the field of polynomials with coefficients in Z_q modulo 
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
        return polymod(operation(self, other), self.qpoly, self.q)
        
    def __add__(self, other):
        if not isinstance(other, GaloisElement):
            return super().__add__(other)
        return self.pairwise_operate(other, "add")

    def __mul__(self, other):
        if not isinstance(other, GaloisElement):
            return super().__mul__(other)
        return self.pairwise_operate(other, "mul")

if __name__ == "__main__":
    a = GaloisElement([1, 0, 1], q=2, m=4)
    b = GaloisElement([1, 0, 0, 0, 1], q=2, m=4)
    print(repr(a))
    print(repr(b))
    print(a + b)
    print(a * b)