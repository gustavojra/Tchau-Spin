from .index import Index
import numpy as np
import copy

class Tensor:

    # General tensor representation, also base class for specific tensor as Fock, ERI and Amplitudes

    def __init__(self, name, *argv, prefac=1, rhf=False):
        self.rhf = rhf
        self.idx = [] 
        for index in argv:
            if type(index) != Index:
                raise TypeError('Indexes cannot be {}'.format(type(index)))
            self.idx.append(index)

        if type(prefac) != int and type(prefac) != float:
            raise TypeError('Prefacor needs to be a number, not a {}'.format(type(prefac)))
        self.prefac = prefac

        if type(name) != str:
            raise TypeError('Name needs to be a string, not a {}'.format(type(name)))
        self.name = name

        self.create_repr()


    def __str__(self):

        if self.prefac == -1:
            return '-' + self.repr

        elif self.prefac == 1:
            return '+' + self.repr

        return str(self.prefac) +'*' + self.repr

    def create_repr(self):

        # Create string representation

        self.repr = str(self.name) + '('
        for i in self.idx:
            self.repr += str(i)
        self.repr += ')'

    def copy(self):

        return Tensor(self.name, *self.idx, prefac=self.prefac, rhf=self.rhf)

    def __add__(self, other):

        # Addition: first put self into a collection, then call collectin addition method
        
        c = Collection() + self

        return c + other

    def __radd__(self, other):

        return self + other

    def __sub__(self, other):

        c = Collection() + self
        
        return c - other

    def __rsub__(self, other):

        c = Collection() - self

        return c + other

    def __mul__(self, other):

        # Multiplication of Tensors will produce a Contraction object
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

            return Contraction.contract(self, other)

        if isinstance(other, Collection):
            return other*self

        if isinstance(other, Contraction):
            if self.any_undef_spin():
                return other*self.expand()
            return other*self

        else:

            raise TypeError('Multiplication undefined for {} and {}'.format(type(self),type(other)))

    def __rmul__(self, other):

        return self*other

    def __pow__(self, other):

        # Power of Tensors a spin-free contraction
        # If the multiplying are not spin defined, expantian will be called first

        if isinstance(other, Tensor):

            if self.any_undef_spin() and other.any_undef_spin():
                return (self.expand())**(other.expand())

            elif self.any_undef_spin():
                return (self.expand())**other

            elif other.any_undef_spin():
                return self**(other.expand())

            return Contraction.spin_free_contract(self, other)

        if isinstance(other, Collection):
            return other**self

        if isinstance(other, Contraction):
            terms = [other.prefac*self] + other.contracting
            return Contraction.spin_free_contract(*terms)

        else:

            raise TypeError('Spin-free Contraction undefined for {} and {}'.format(type(self),type(other)))
    
    def __eq__(self, other):

        # Two tensor are equal if they share the same type and indexes 

        if type(self) != type(other):
            return False

        if len(self.idx) != len(other.idx):
            return False

        for s,o in zip(self.idx, other.idx):
            if s != o:
                return False

        return True

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
        out = Collection()
        while not it.finished:
            if it[0] > 0:
                newidx = []
                for i in range(len(it.multi_index)):
                    newidx.append(self.idx[i].change_spin(it.multi_index[i]))
                out += Tensor(self.name, *newidx, prefac=self.prefac)
            it.iternext()
    
        return out

    def flip(self):

        # Return a copy of the tensor where all spin indexes are inverted

        newidx = []
        for i in self.idx:
            newidx.append(i.flip())
        out = self.copy()
        out.idx = newidx
        out.create_repr()
        return out

class Fock(Tensor):

    # Object that represents a fock matrix for restricted orbitals

    def __init__(self, index_1, index_2, prefac=1, rhf=False):

        self.rhf = rhf
        for index in [index_1, index_2]:
            if type(index) != Index:
                raise TypeError('Indexes cannot be {}'.format(type(index)))
        self.idx = [index_1, index_2]
        self.prefac = prefac
        self.create_repr()

    def copy(self):

        # Return a copy of itself

        return Fock(*self.idx, prefac=self.prefac, rhf=self.rhf)

    def create_repr(self):

        # Create a string representation of Fock 

        self.repr = 'f(' + str(self.idx[0]) + ',' + str(self.idx[1]) + ')'

    def expand(self):

        # If spin-orbital indexes exist, expand all valid spin cases

        possible = self.spin_cases()
        
        # For fock matrix f(p,q) spin of p has to be equal spin q. Thus

        out = Collection()
        for p in range(2):
            for q in range(2):
                if possible[p,q] == 1:
                    if p == q:
                        ip = self.idx[0].change_spin(p)
                        iq = self.idx[1].change_spin(q)
                        out += Fock(ip, iq, prefac=self.prefac, rhf=self.rhf)

        return out

    def adapt(self):

        # If RHF is true and Fock is beta,beta return alpha,alpha

        if self.any_undef_spin():
            raise NameError('Cannot adapt while indexes have undefined spin')
        if self.idx[0].spin == 'beta' and self.idx[1].spin == 'beta' and self.rhf:
            return self.flip()
        else:
            return self.copy()

class ERI(Tensor):

    # Special tensor for the Two-electron integral array. Indices in Physicist's notation

    @staticmethod
    def Antisymmetric(p,q,r,s, prefac=1, rhf=False):

        # Return the antisymmetric tensor ERI as a combination of two regular ERI:
        # <pq||rs> = <pq|rs> - <pq|sr>

        return ERI(p,q,r,s, prefac=prefac, rhf=rhf) - ERI(p,q,s,r, prefac=prefac, rhf=rhf)

    def __init__(self, p, q, r, s, prefac=1, rhf=False):

        self.rhf = rhf
        self.idx = [p,q,r,s]
        for index in self.idx:    
            if type(index) != Index:
                raise TypeError('Indexes cannot be {}'.format(type(index)))
        self.prefac = prefac      
        self.create_repr()

    def copy(self):

        # Return a copy of itself

        return ERI(*self.idx, prefac=self.prefac, rhf=self.rhf)

    def create_repr(self):

        # Create a string representation of ERI

        [p,q,r,s] = self.idx
        self.repr = '<' +  str(p) + str(q) + '|' + str(r) + str(s) + '>'

    def expand(self):

        possible = self.spin_cases()
        
        # For the ERI <pq|rs> we have the spin condition p=r and q=s

        out = Collection()
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
                                out += ERI(a,b,c,d, prefac=self.prefac, rhf=self.rhf)
        return out

    def adapt(self):
    
        # If all indexes are have beta spin and rhf is true, then flip spin
        if self.any_undef_spin():
            raise NameError('Cannot adapt while indexes have undefined spin')
        nalpha = np.sum(np.array([x.s for x in self.idx]))
        if nalpha == 0 and self.rhf:
            return self.flip()
        else:
            return self.copy()

class Amplitude(Tensor):

    # Special tensor to represent coupled cluster amplitudes.
    # First half of index are taken as occupied indexes (below Fermi), second half as virtual indexes (above Fermi)

    def __init__(self, *argv, prefac=1, rhf=False):

        self.rhf = rhf
        self.idx = [] 
        for index in argv:
            if type(index) != Index:
                raise TypeError('Indexes cannot be {}'.format(type(index)))
            self.idx.append(index)

        if len(self.idx) % 2 != 0:
            raise NameError('Amplitudes cannot have odd number of indexes')

        self.rank = int(len(self.idx)/2)
        self.prefac = prefac
        self.create_repr()


    def copy(self):

        # Returna  copy of itself

        return Amplitude(*self.idx, prefac=self.prefac, rhf=self.rhf)

    def create_repr(self):

        # Create string representation of Amplitude

        self.repr = 'T{}('.format(self.rank)
        for n,i in enumerate(self.idx):
            if n == self.rank:
                self.repr += '->'
            self.repr += str(i)
        self.repr += ')'

    def expand(self):

        possible = self.spin_cases()
        
        # For amplitude, we require that the number of alphas/betas in the occupied indexes equals the number of alpha/betas in the virtual space
        # e.g. Ij->Ab is allowed, but IJ->ab is not

        it = np.nditer(possible, flags=['multi_index'])
        out = Collection()
        l = self.rank
        while not it.finished:
            if it[0] > 0:
                occ = it.multi_index[0:l]
                vir = it.multi_index[l:]
                if sum(occ) == sum(vir):
                    newidx = []
                    for i in range(len(it.multi_index)):
                        newidx.append(self.idx[i].change_spin(it.multi_index[i]))
                    out += Amplitude(*newidx, prefac=self.prefac, rhf=self.rhf)
            it.iternext() 

        return out

    def adapt(self):

        # Adapt amplitudes to a stardard form
        # This is used to try to reduce the number of spin cases to be considered

        # First check if all spin indexes are defined. If not, raise an error
        if self.any_undef_spin():
            raise NameError('Cannot adapt while indexes have undefined spin')

        # Get index for hole and particle spaces
        holes = copy.deepcopy(self.idx[0:self.rank])
        par = copy.deepcopy(self.idx[self.rank:])

        # Get number of alpha and beta indexes
        nalpha = np.sum(np.array([x.s for x in self.idx]) == 1)
        nbeta = np.sum(np.array([x.s for x in self.idx]) == 0)
        
        # If number of alphas is greater than alpha and RHF is True perform spin flip
        # This allow us to have the maximum number of amplitudes in a standard form
        if nbeta > nalpha and self.rhf:
            return self.flip().adapt()

        # If the Amplitude is composed of all alpha indexes and RHF is true, then
        # we can write it as a linear combination of mixed spin cases
        # e.g. IJ->AB   can be expressed as Ij->Ab + iJ->Ab
        # e.g. IJK->ABC can be expressed as IjK->AbC + iJK->AbC + IJk->AbC
        # Note that every term will be adapted again, thus the results won't actuall match the example above

        if nbeta == 0 and self.rank > 1 and self.rhf:
            out = Collection()
            par[1] = par[1].flip()
            for i,h in enumerate(holes):
                newholes = holes[:]
                newholes[i] = h.flip()
                newidx = newholes + par
                out += Amplitude(*newidx, prefac=self.prefac, rhf=self.rhf).adapt()
            return out

        # If the two tests above fail, then we have a mixed spin case.
        # The goal here is to reorder the amplitude indexes following antisymmetric rules
        # such that we have every amplitude in a standard form.
        # I chose the standard form as alphas and betas alternated. If the number fo alphas
        # and betas are not the same, the first indexes will be alternating and the final ones
        # will be the remaining indexes
        # e.g. iJ->Ab   becomes (-1)*Ji->Ab
        # e.g. IJk->aBC becomes      IkJ->BaC
        # e.g. Ijk->aBc becomes (-1)*Ijk->Bac
        
        factor = 1
        new_holes = []
        new_par = []
        while len(holes) > 0:
            # Search for alpha
            perm = 0
            for h in holes:
                if h.s == 1:
                    new_holes.append(h)
                    factor *= (-1)**perm
                    holes.remove(h)
                    break
                perm += 1
            # Search for beta
            perm = 0
            for h in holes:
                if h.s == 0:
                    new_holes.append(h)
                    factor *= (-1)**perm
                    holes.remove(h)
                    break
                perm += 1

        while len(par) > 0:
            # Search for alpha
            perm = 0
            for p in par:
                if p.s == 1:
                    new_par.append(p)
                    factor *= (-1)**perm
                    par.remove(p)
                    break
                perm += 1
            # Search for beta
            perm = 0
            for p in par:
                if p.s == 0:
                    new_par.append(p)
                    factor *= (-1)**perm
                    par.remove(p)
                    break
                perm += 1
    
        idx = new_holes + new_par
        return Amplitude(*idx, prefac = self.prefac*factor, rhf=self.rhf)

class Collection:

    # Object to represent a sum of terms and Contractions

    def __init__(self, terms=[]):

        self.terms = terms

    # Define addition operations

    def __add__(self, other):

        # If 0 is added, return itself
        if other == 0:
            return Collection(terms=self.terms)

        elif isinstance(other, Tensor) or isinstance(other, Contraction):
            # If other is a tensor or Contraction append it to terms
            newterms = self.terms[:]
            newterms.append(other)
            return Collection(newterms)

        elif type(other) == Collection:
            # For two Contractions, merge the terms
            newterms = self.terms[:] + other.terms[:]
            return Collection(newterms)

        else:
            return TypeError('Addition undefined for collection and {}'.format(type(other)))

    def __radd__(self, other):

        return self + other

    def __sub__(self, other):

        # If 0 is added, return itself
        if other == 0:
            return Collection(terms=self.terms)

        elif isinstance(other, Tensor) or isinstance(other, Contraction):
            # If other is a tensor or Contraction append it to
            newother = other.copy()
            newother.prefac *= -1
            newterms = self.terms[:]
            newterms.append(newother)
            return Collection(newterms)

        elif isinstance(other, Collection):
            newterms = other.terms[:]
            for X in newterms:
                X.prefac *= -1
            newterms = self.terms[:] + newterms
            return Collection(newterms)

    # Iteration thorugh a collection goes through each term

    def __iter__(self):

        return iter(self.terms)

    # Define multiplication

    def __mul__(self, other):

        if isinstance(other, Collection):
            out = Collection()
            for Y in self:
                for X in other:
                    out += Y*X
            return out

        else:
            out = Collection()
            for X in self:
                out += other*X
            return out

    def __rmul__(self, other):

        return self*other

    def __pow__(self, other):

        if isinstance(other, Collection):
            out = Collection()
            for Y in self:
                for X in other:
                    out += Y**X
            return out

        else:
            out = Collection()
            for X in self:
                out += other**X
            return out

    def __str__(self):

        out  = ''
        for x in self.terms:
            out += str(x) + '   '

        return out

    def __len__(self):
        return len(self.terms)

    def expand(self):
        out = Collection()
        for X in self:
            if isinstance(X, Tensor):
                out += X.expand()
            else:
                out += X
        return out

    def adapt(self):
        out = Collection()
        for X in self:
            out += X.adapt()
        return out

class Contraction:

    @staticmethod
    def contract(*argv):

        # Static contructor function for contrations
        for A in argv:
            if not isinstance(A, Tensor):
                raise TypeError('Contraction not defined for {}'.format(type(A)))
            if A.any_undef_spin():
                raise NameError('Tensor spin-orbital indexes cannot be contracted: {}'.format(A))

        names = []
        indexes = []
        for A in argv:
            names += [x.name for x in A.idx]
            indexes += A.idx

        internal = []
        external = []
        for i in indexes:
            if np.sum(np.array(indexes) == i) > 1:
                if i not in internal:
                    internal.append(i)
            else:
                external.append(i)

        # If one of the contracting tensors does not have a index contracting, return 0
        for A in argv:
            U = np.intersect1d([x for x in A.idx], [x for x in internal])
            if len(U) == 0:
                return 0

        for i in external:
            if i.flip() in external:
                return 0

        return Contraction(*argv, inter = internal, ext = external)
            
    @staticmethod
    def spin_free_contract(*argv):

        # Contract two tensor without analysing valid spin cases
        # Be careful using this!!
        for A in argv:
            if not isinstance(A, Tensor):
                raise TypeError('Contraction not defined for {}'.format(type(A)))

        names = []
        indexes = []
        for A in argv:
            names += [x.name for x in A.idx]
            indexes += A.idx

        internal = []
        external = []
        # All indexes are saved as alpha for a spinless contraction
        for i in indexes:
            if names.count(i.name) > 1:
                if i.alpha() not in internal:
                    internal.append(i.alpha())
            else:
                external.append(i.alpha())

        # If one of the contracting tensors does not have a index contracting, return 0
        for A in argv:
            U = np.intersect1d([x.name for x in A.idx], [x.name for x in internal])
            if len(U) == 0:
                return 0
        else:
            return Contraction(*argv, inter=internal, ext=external, spin_free=True)

    def __init__(self, *argv, inter, ext, prefac=1, spin_free=False):

        self.int = inter
        self.ext = ext
        self.prefac = prefac
        self.contracting = list(argv)
        self.sort()
        self.spin_free = spin_free
        for A in self.contracting:
            self.prefac *= A.prefac
            A.prefac = 1

    def sort(self):
        
        # Order terms in the contracting according to rules:
        # 1) First fock elements
        # 2) Second amplitudes, from lower to higer ranks
        # 3) Third, general Tensors other than ERI
        # 4) ERI

        sortkey = []
        for c in self.contracting:
            if type(c) == Fock:
                sortkey.append('a')
            elif type(c) == Amplitude:
                sortkey.append('b' + str(c.rank))
            elif type(c) == Tensor:
                sortkey.append('c')
            elif type(c) == ERI:
                sortkey.append('d')
            else:
                sortkey.append('e')
        sortkey = np.array(sortkey)
        key = sortkey.argsort()
        self.contracting = list(np.array(self.contracting)[key])

    def __str__(self):

        out = str(self.prefac)
        for X in self.contracting:
            out += '*' + X.repr

        return out

    def __mul__(self, other):

        if type(other) == float or type(other) == int:
            return Contraction(*self.contracting, inter=self.int, ext=self.ext, prefac=self.prefac*other)

        if isinstance(other, Tensor):
            if self.spin_free:
                raise NameError('Cannot further contract a Spin-Free contraction (adapted). Try reorder your operations')
            C = self.contracting + [other]
            return Contraction.contract(*C)

        if isinstance(other, Collection):
            return other*self

        else:
            raise TypeError('Multiplication not defined for Contraction and {}'.format(type(other)))

    def __rmul__(self, other):

        return self*other

    def adapt(self):

        out = Collection()
        out += self.contracting[0].adapt()
        for c in self.contracting[1:]:
            out = out**(c.adapt())
        return self.prefac*out
