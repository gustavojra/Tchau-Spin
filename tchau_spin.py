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

        elif isinstance(other, Tensor):

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


class Tensor:

    # General tensor representation, also base class for specific tensor as Fock, ERI and Amplitudes

    def __init__(self, name, *argv, prefac=1):
        self.idx = [] 
        for index in argv:
            self.idx.append(index)

        self.prefac = prefac
        self.name = name

    def copy(self):

        return Tensor(self.name, *self.idx, prefac=self.prefac)

    def __add__(self, other):

        # Addition: first put self into a collection, then call collectin addition method
        
        c = collection() + self

        return c + other

    def __mul__(self, other):

        if type(other) == float or type(other) == int:

            out = self.copy()
            out.prefac = out.prefac*other
            return out

        else:

            raise TypeError('Multiplication undefined for {} and {}'.format(type(self),type(other)))

    def __rmul__(self, other):

        return self*other

    def __str__(self):

        out = str(self.prefac) + '*' + str(self.name) + '('
        for i in self.idx:
            out += str(i)
        out += ')'

        return out

class fock_rhf(Tensor):

    # Object that represents a fock matrix for restricted orbitals

    def __init__(self, index_1, index_2, prefac=1):

        self.idx = [index_1, index_2]
        self.prefac = prefac

    def copy(self):

        return fock_rhf(*self.idx, prefac=self.prefac)

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
                    out += fock_rhf(ip, iq, prefac=self.prefac)

        return out
                    
    def spin_simplify(self):

        if any([x.s == 2 for x in self.idx]):
            
            raise NameError('Cannot simplify with spin orbital indexes')

        else:

            pass

class ERI(Tensor):

    def __init__(self, p, q, r, s, prefac=1):

        self.idx = [p,q,r,s]
        self.prefac = prefac

    def copy(self):

        return ERI(*self.idx, prefac=self.prefac)

    def __str__(self):

        [p,q,r,s] = self.idx

        return str(self.prefac) + '*<' +  str(p) + str(q) + '|' + str(r) + str(s) + '>'

    def expand_spin(self):

        # Array with all possible combinations os alpha and beta for each index
        # One represent an allowed spin case
        possible = np.ones((2,2,2,2))
        
        # For the ERI <pq|rs> we have the spin condition p=r and q=s
        # Note: this part could be hardcoded, since its the same for every ERI

        for p in range(2):
            for q in range(2):
                for r in range(2):
                    for s in range(2):
                        if p != r or q != s:
                            possible[p,q,r,s] = 0

        # Check if any index has spin defined
        # If so, deleted spin cases that dont match it
        for i,x in enumerate(self.idx):

            sl = [slice(0,2), slice(0,2), slice(0,2), slice(0,2)]
            if x.spin == 'alpha':
                sl[i] = 0
                possible[sl[0], sl[1], sl[2], sl[3]] = 0
            elif x.spin == 'beta':
                sl[i] = 1
                possible[sl[0], sl[1], sl[2], sl[3]] = 0

        # Loop through remaining and create return collection of spin cases
        out = collection()
        for p in range(2):
            for q in range(2):
                for r in range(2):
                    for s in range(2):
                        if possible[p,q,r,s] == 1:
                            a = self.idx[0].change_spin(p)
                            b = self.idx[1].change_spin(q)
                            c = self.idx[2].change_spin(r)
                            d = self.idx[3].change_spin(s)
                            out += ERI(a,b,c,d, prefac=self.prefac)

        return out
                
class Amplitude(Tensor):

    def __init__(self, *argv, prefac=1):

        self.idx = [] 
        for index in argv:
            self.idx.append(index)

        if len(self.idx) % 2 != 0:
            raise NameError('Amplitudes cannot have odd number of indexes')

        self.rank = int(len(self.idx)/2)
        self.prefac = prefac

    def copy(self):

        return Amplitude(*self.idx, prefac=self.prefac)

    def __str__(self):

        hole = self.idx[0:self.rank]
        par = self.idx[self.rank:]

        out = str(self.prefac) + 'T('
        for h in hole:
            out += str(h)
        out += ','
        for p in par:
            out += str(p)
        out += ')'

        return out



