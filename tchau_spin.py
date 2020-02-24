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

    def __eq__(self, other):

        # Two indexes are the sme if name and spin match

        if type(other) == Index:
            return self.name == other.name and self.spin == other.spin
        return False

    def change_spin(self, s):

        # Return a copy of the index with the desired spin (0 for beta, 1 for alpha)

        if s == 1:
            return Index(self.name, spin='alpha')

        elif s == 0:
            return Index(self.name, spin='beta')

        else:
            raise ValueError('Invalid spin input. Must be 0 (for beta) or 1 (for alpha)')

    def flip(self):

        # Return copy with opposite spin

        if self.spin == 'alpha':
            return Index(self.name, spin='beta')
        elif self.spin == 'beta':
            return Index(self.name, spin='alpha')
        else:
            raise NameError('Cannot flip undefined spin')

class collection:

    # Object to represent a sum of terms and contractions

    def __init__(self, terms=[]):

        self.terms = terms

    # Define addition operations

    def __add__(self, other):

        # If 0 is added, return itself
        if other == 0:
            return collection(terms=self.terms)

        elif isinstance(other, Tensor) or isinstance(other, contraction):
            # If other is a tensor or contraction append it to terms
            newterms = self.terms[:]
            newterms.append(other)
            return collection(newterms)

        elif type(other) == collection:
            # For two contractions, merge the terms
            newterms = self.terms[:] + other.terms[:]
            return collection(newterms)

        else:
            return TypeError('Addition undefined for collection and {}'.format(type(other)))

    def __radd__(self, other):

        return self + other

    # Iteration thorugh a collection goes through each term

    def __iter__(self):

        return iter(self.terms)

    # Define multiplication

    def __mul__(self, other):

        if type(other) == float or type(other) == int:
            out = collection()
            for X in self:
                out += X*other 
            return out

        if isinstance(other, collection):
            out = collection()
            for Y in self:
                for X in other:
                    out += Y*X
            return out

    def __rmul__(self, other):

        return self*other

    def __str__(self):

        out  = ''
        for x in self.terms:
            out += str(x) + '   '

        return out

class contraction:

    @staticmethod
    def contract(A,B):

        # Static contructor function for contrations

        if isinstance(A, Tensor) and isinstance(B, Tensor):
            if A.any_undef_spin() or B.any_undef_spin():
                raise NameError('Tensor with undefined spin cannot be contracted')

            internal = []
            external = []
            for i in A.idx + B.idx:
                if i in A.idx and i in B.idx:
                    if i not in internal:
                        internal.append(i)
                else:
                    external.append(i)
            if len(internal) == 0:
                return 0
            for i in external:
                if i.flip() in external:
                    return 0
            else:
                return contraction(A, B, inter=internal, ext=external, prefac=A.prefac*B.prefac)
            
        else:
            raise TypeError('Contraction not defined for {} and {}'.format(type(A), type(B)))

    def __init__(self, *argv, inter, ext, prefac=1):

        self.int = inter
        self.ext = ext
        self.prefac = prefac
        self.contracting = list(argv)

    def __str__(self):

        out = str(self.prefac)
        for X in self.contracting:
            out += '*' + X.repr

        return out

    def __mul__(self, other):

        if type(other) == float or type(other) == int:
            return contraction(*self.contracting, inter=self.int, ext=self.ext, prefac=self.prefac*other)

    def __rmul__(self, other):

        return self*other

class Tensor:

    # General tensor representation, also base class for specific tensor as Fock, ERI and Amplitudes

    def __init__(self, name, *argv, prefac=1):
        self.idx = [] 
        for index in argv:
            self.idx.append(index)

        if type(prefac) != int and type(prefac) != float:
            raise TypeError('Prefacor needs to be a number, not a {}'.format(type(prefac)))
        self.prefac = prefac

        if type(name) != str:
            raise TypeError('Name needs to be a string, not a {}'.format(type(name)))
        self.name = name

        # Create string representation
        self.repr = str(self.name) + '('
        for i in self.idx:
            self.repr += str(i)
        self.repr += ')'

    def __str__(self):

        if self.prefac == -1:
            return '-' + self.repr

        elif self.prefac == 1:
            return '+' + self.repr

        return str(self.prefac) +'*' + self.repr

    def copy(self):

        return Tensor(self.name, *self.idx, prefac=self.prefac)

    def __add__(self, other):

        # Addition: first put self into a collection, then call collectin addition method
        
        c = collection() + self

        return c + other

    def __radd__(self, other):

        return self + other

    def __mul__(self, other):

        # Multiplication of Tensors will produce a contraction object
        # If the multiplying are not spin defined, expantian will be called first

        if type(other) == float or type(other) == int:

            out = self.copy()
            out.prefac = out.prefac*other
            return out

        if isinstance(other, Tensor):

            if self.any_undef_spin() and other.any_undef_spin():
                return (self.expand())*(other.expand())

            elif self.any_undef_spin():
                return (self.expand())*other

            elif other.any_undef_spin():
                return self*(other.expand())

            return contraction.contract(self, other)

        if isinstance(other, collection):
            out = collection()
            for X in other:
                out += self*X
            return out

        else:

            raise TypeError('Multiplication undefined for {} and {}'.format(type(self),type(other)))

    def __rmul__(self, other):

        return self*other

    def any_undef_spin(self):

        # Check if any index is spin undefined

        x = np.array([s.spin for s in self.idx]) == 'any'
        return any(x)

    def spin_cases(self):

        # Return a N dimentional array (N = number of indexes) where each array ranges from 0,1 representing beta and alpha spins
        # For each entry, 1 indicates that that spin combination is allowed, 0 indicates that that spin combination is not allowed
        # For this general tensor, a case is allowed if its not in conflicte with the Indexes that have spin defined

        possible = np.ones((2,)*len(self.idx))

        # Check if any index has spin defined
        # If so, deleted spin cases that dont match it
        for i,x in enumerate(self.idx):

            sl = [slice(0,2)]*len(self.idx)
            if x.spin == 'alpha':
                sl[i] = 0
                possible[tuple(sl)] = 0
            elif x.spin == 'beta':
                sl[i] = 1
                possible[tuple(sl)] = 0

        return possible

    def expand(self):

        # Expand the tensor in each allowed spin case

        possible = self.spin_cases()
        it = np.nditer(possible, flags=['multi_index'])
        out = collection()
        while not it.finished:
            if it[0] > 0:
                newidx = []
                for i in range(len(it.multi_index)):
                    newidx.append(self.idx[i].change_spin(it.multi_index[i]))
                out += Tensor(self.name, *newidx, prefac=self.prefac)
            it.iternext()
    
        return out

class fock(Tensor):

    # Object that represents a fock matrix for restricted orbitals

    def __init__(self, index_1, index_2, prefac=1):

        self.idx = [index_1, index_2]
        self.prefac = prefac
        self.repr = 'f(' + str(self.idx[0]) + ',' + str(self.idx[1]) + ')'

    def copy(self):

        return fock(*self.idx, prefac=self.prefac)

    def expand(self):

        possible = self.spin_cases()
        
        # For fock matrix f(p,q) spin of p has to be equal spin q. Thus

        out = collection()
        for p in range(2):
            for q in range(2):
                if possible[p,q] == 1:
                    if p == q:
                        ip = self.idx[0].change_spin(p)
                        iq = self.idx[1].change_spin(q)
                        out += fock(ip, iq, prefac=self.prefac)

        return out
                    
class ERI(Tensor):

    # Special tensor for the Two-electron integral array. Indices in Physicist's notation

    def __init__(self, p, q, r, s, prefac=1):

        self.idx = [p,q,r,s]
        self.prefac = prefac
        self.repr = '<' +  str(p) + str(q) + '|' + str(r) + str(s) + '>'

    def copy(self):

        return ERI(*self.idx, prefac=self.prefac)

    def expand(self):

        possible = self.spin_cases()
        
        # For the ERI <pq|rs> we have the spin condition p=r and q=s

        out = collection()
        for p in range(2):
            for q in range(2):
                for r in range(2):
                    for s in range(2):
                        if possible[p,q,r,s] == 1:
                            if p == r and q == s:
                                a = self.idx[0].change_spin(p)
                                b = self.idx[1].change_spin(q)
                                c = self.idx[2].change_spin(r)
                                d = self.idx[3].change_spin(s)
                                out += ERI(a,b,c,d, prefac=self.prefac)
        return out
                
class Amplitude(Tensor):

    # Special tensor to represent coupled cluster amplitudes.
    # First half of index are taken as occupied indexes, second half as virtual indexes

    def __init__(self, *argv, prefac=1):

        self.idx = [] 
        for index in argv:
            self.idx.append(index)

        if len(self.idx) % 2 != 0:
            raise NameError('Amplitudes cannot have odd number of indexes')

        self.rank = int(len(self.idx)/2)
        self.prefac = prefac

        # Create string representation
        self.repr = 'T{}('.format(self.rank)
        for n,i in enumerate(self.idx):
            if n == self.rank:
                self.repr += '->'
            self.repr += str(i)
        self.repr += ')'

    def copy(self):

        return Amplitude(*self.idx, prefac=self.prefac)

    def expand(self):

        possible = self.spin_cases()
        
        # For amplitude, we require that the number of alphas/betas in the occupied indexes equals the number of alpha/betas in the virtual space
        # e.g. Ij->Ab is allowed, but IJ->ab is not

        it = np.nditer(possible, flags=['multi_index'])
        out = collection()
        l = self.rank
        while not it.finished:
            if it[0] > 0:
                occ = it.multi_index[0:l]
                vir = it.multi_index[l:]
                if sum(occ) == sum(vir):
                    newidx = []
                    for i in range(len(it.multi_index)):
                        newidx.append(self.idx[i].change_spin(it.multi_index[i]))
                    out += Amplitude(*newidx, prefac=self.prefac)
            it.iternext() 

        return out
