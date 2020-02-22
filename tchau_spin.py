import numpy as np

class Index:

    # Class to represente indexes using in quantum chemistry calculations
    # Must have a name (e.g. i,j,k,l,m,n, etc)
    # Msut have a spin (alpha, beta or any)
    
    def __init__(self, name, spin='any'):

        if len(name) > 1:
            raise NameError('Index must be identified by one letter')
        self.name = name
        self.spin = spin
        if spin == 'alpha':
            self.s = 1
        elif spin == 'beta':
            self.s = 0
        elif spin =='any':
            self.s = 2
        else:
            raise NameError('Invalid Spin Type. Must be alpha, beta or any')

    def __str__(self):

        # String representation of the index:
        # Alpha indexes are capitalized
        # Beta indexes are lower cases
        # Spin indefined index are represented with '~' before it

        if self.s == 1:
            return self.name.upper()
        elif self.s == 0:
            return self.name.lower()
        else:
            return '~'+self.name

    def change_spin(self, s):

        # Return a copy of the index with the desired spin (0 for beta, 1 for alpha)

        if s == 1:
            return Index(self.name, spin='alpha')

        elif s == 0:
            return Index(self.name, spin='beta')

        else:
            raise ValueError('Invalid spin input. Must be 0 (for beta) or 1 (for alpha)')

class collection:

    # Object to represent a sum of terms and contractions

    def __init__(self, terms=[]):

        self.terms = terms

    # Define addition operations

    def __add__(self, other):

        if other == 0:
            return self

        elif type(other) == fock_rhf:

            newterms = self.terms[:]
            newterms.append(other)
            return collection(newterms)

        elif type(other) == collection:
            newterms = self.terms[:] + other.terms[:]
            return collection(newterms)

        else:
            return TypeError('Addition undefined for collection and {}'.format(type(other)))

    def __radd__(self, other):

        return self + other

    def __str__(self):

        out  = ''
        for x in self.terms:
            out += str(x) + '   '

        return out


class fock_rhf:

    # Object that represents a fock matrix for restricted orbitals

    def __init__(self, index_1, index_2, prefac=1):

        self.idx = [index_1, index_2]
        self.prefac = prefac


    def __add__(self, other):

        # Addition: first put self into a collection, then call collectin addition method
        
        c = collection() + self

        return c + other

    def __mul__(self, other):

        if type(other) == float or type(other) == int:
            
            return fock_rhf(self.idx[0], self.idx[1], prefac = self.prefac*other)

        else:

            raise TypeError('Addition undefined for fock and {}'.format(type(other)))

    def __rmul__(self, other):

        return self*other

    def __str__(self):

        if self.prefac == -1:
            return '-f(' + str(self.idx[0]) + ',' + str(self.idx[1]) + ')'

        elif self.prefac == 1:
            return '+f(' + str(self.idx[0]) + ',' + str(self.idx[1]) + ')'

        return str(self.prefac) +'*f(' + str(self.idx[0]) + ',' + str(self.idx[1]) + ')'

    def expand_spin(self):

        # Array with all possible combinations os alpha and beta for each index
        # One represent an allowed spin case
        possible = np.ones((2,2))
        
        # For fock matrix f(p,q) spin of p has to be equal spin q. Thus

        for p in range(2):
            for q in range(2):
                if p != q:
                    possible[p,q] = 0

        # Check if any index has spin defined:
        if self.idx[0].spin == 'beta':
            possible[1,:] = 0

        if self.idx[0].spin == 'alpha':
            possible[0,:] = 0

        if self.idx[1].spin == 'beta':
            possible[:,1] = 0

        if self.idx[1].spin == 'alpha':
            possible[:,0] = 0

        out = collection()
        
        for p in range(2):
            for q in range(2):
                if possible[p,q] == 1:
                    ip = self.idx[0].change_spin(p)
                    iq = self.idx[1].change_spin(q)
                    out += fock_rhf(ip, iq)

        return out
                    
    def spin_simplify(self):

        if any([x.s == 2 for x in self.idx]):
            
            raise NameError('Cannot simplify with spin orbital indexes')

        else:

            pass

class ERI:

    def __init__(p, q, r, s, prefac):

        self.idx = [p,q,r,s]
        self.prefac = 1.0
                
#class Amplitude:
#
#    def __init__(self, name, 
#


