from .index import Index
import numpy as np
import copy

class Tensor:

    # RHF option. If true it allows more transformations (such as spin reflection)  when the function 'adapt' is called
    rhf = False

    # General tensor representation, also base class for specific tensor as Fock, ERI and Amplitudes

    def __init__(self, name, *argv):

        # Requires a name (string)
        # Requires indexes (*argv)

        self.idx = [] 
        for index in argv:
            if type(index) != Index:
                raise TypeError('Indexes cannot be {}'.format(type(index)))
            self.idx.append(index)

        if type(name) != str:
            raise TypeError('Name needs to be a string, not a {}'.format(type(name)))
        self.name = name
        self.create_repr()

    def __str__(self):

        # Return the string representation previously created

        return self.repr

    def create_repr(self):

        # Create string representation to be used when printing

        self.repr = str(self.name) + '('
        for i in self.idx:
            self.repr += str(i)
        self.repr += ')'

    def copy(self):

        # Return a copy of itself

        return Tensor(self.name, *self.idx)

    def __add__(self, other):

        # Additions are performed through the Collection object
        
        c = Collection() + self

        return c + other

    def __radd__(self, other):

        # Additions are performed through the Collection object

        return self + other

    def __sub__(self, other):

        # Subtractions are performed through the Collection object

        c = Collection() + self
        
        return c - other

    def __rsub__(self, other):

        # Subtractions are performed through the Collection object

        c = Collection() - self

        return c + other

    def __mul__(self, other):

        # Multiplication of Tensors will produce a Contraction object

        if type(other) == float or type(other) == int:

            # Multiplication by an scalar is performed through the Collection object

            c = Collection() + self
            return c*other

        if isinstance(other, Tensor):

            # If the multiplying Tensors have spin-orbital indexes, 'expand' will be called first

            if self.any_undef_spin() and other.any_undef_spin():
                return (self.expand())*(other.expand())

            elif self.any_undef_spin():
                return (self.expand())*other

            elif other.any_undef_spin():
                return self*(other.expand())

            return Contraction.contract(self, other)

        # Multiplication by Collection or Contraction object are implemented in the respective class

        if isinstance(other, Collection):
            return Collection.__mul__(other, self)

        if isinstance(other, Contraction):
            return Contraction.__mul__(other, self)

        else:

            raise TypeError('Multiplication undefined for {} and {}'.format(type(self),type(other)))

    def __rmul__(self, other):

        return self*other

    def __pow__(self, other):

        # The Power operator (**) represents a spin-free contraction, that is, a contraction
        # where the spin compatibility is not ensured. Use carefully.

        if isinstance(other, Tensor):

            # If the multiplying Tensors have spin-orbital indexes, 'expand' will be called first

            if self.any_undef_spin() and other.any_undef_spin():
                return (self.expand())**(other.expand())

            elif self.any_undef_spin():
                return (self.expand())**other

            elif other.any_undef_spin():
                return self**(other.expand())

            return Contraction.spin_free_contract(self, other)

        # Power with Collection or Contraction object are implemented in the respective class

        if isinstance(other, Collection):
            return Collection.__pow__(other, self)

        if isinstance(other, Contraction):
            return Contraction.__pow__(other, self)

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

        # Check if any index is spin-orbital

        x = np.array([s.spin for s in self.idx]) == 'any'
        return any(x)

    def spin_cases(self):

        # Return a N dimentional array (N = number of indexes) where each array ranges from 0,1 representing beta and alpha spins
        # For each entry, 1 indicates that that spin combination is allowed, 0 indicates that that spin combination is not allowed
        # For this general tensor, a case is allowed if it does not conflict with the Indexes that have spin defined already

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

        # The next lines use a numpy trick to iterate thorugh all elements of an arbitrary shape array
        it = np.nditer(possible, flags=['multi_index'])
        out = Collection()
        while not it.finished:
            if it[0] > 0:
                newidx = []
                for i in range(len(it.multi_index)):
                    newidx.append(self.idx[i].change_spin(it.multi_index[i]))
                out += Tensor(self.name, *newidx)
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

    def space(self):

        # Return a string representing the space of each index in the tensor.
        # e.g. for a tensor with indexes i,j,a,b return 'oovv'. (Assuming i,j are occupied indexes and a,b are virtual ones)

        space_string = ''
        for i in self.idx:
            if i.hole:
                space_string += 'o' 
            elif i.particle:
                space_string += 'v'
            else:
                raise NameError('Index {} does not have a space defined'.format(i))

        return space_string

class Fock(Tensor):

    # Object that represents a Fock matrix 

    def __init__(self, index_1, index_2):

        # It takes two indexes as arguments

        for index in [index_1, index_2]:
            if type(index) != Index:
                raise TypeError('Indexes cannot be {}'.format(type(index)))
        self.idx = [index_1, index_2]
        self.create_repr()

    def copy(self):

        # Return a copy of itself

        return Fock(*self.idx)

    def create_repr(self):

        # Create a string representation of Fock 

        self.repr = 'f(' + str(self.idx[0]) + ',' + str(self.idx[1]) + ')'

    def expand(self):

        # If spin-orbital indexes exist, return a Collection with all valid spin cases

        possible = self.spin_cases()
        
        # For fock matrix f(p,q) spin of p has to be equal spin q. Thus

        out = Collection()
        for p in range(2):
            for q in range(2):
                if possible[p,q] == 1:
                    if p == q:
                        ip = self.idx[0].change_spin(p)
                        iq = self.idx[1].change_spin(q)
                        out += Fock(ip, iq)

        return out

    def adapt(self):

        # If RHF is true and Fock is (beta, beta) return (alpha,alpha)

        if self.any_undef_spin():
            raise NameError('Cannot adapt while indexes have undefined spin')

        if self.idx[0].spin == 'beta' and self.idx[1].spin == 'beta' and self.rhf:
            return self.flip()
        else:
            return self.copy()

    def adapt_space(self, verbose = False):

        # Use symmetry properties to put the Fock matrix in a standand form with respect to occupancy space
        # i.e. vo is transformed into ov. Spin is not considered.

        if verbose: print('\nSpace adapting {}'.format(self))
        space_string = self.space()

        if verbose: print('Current Space String: {}'.format(space_string))

        # If the Fock is alredy in standard form, return a copy
        if space_string in ['oo', 'vv', 'ov']:
            if verbose: print('Already in standard form')
            return self.copy()

        # If not, expand in all possible forms and search for the first one that is in standard form 
        else:
            if space_string == 'vo':
                p, q = self.idx
                if verbose: print('Flipping indices!')
                return Fock(q, p)
            else:
                raise NameError('No standard form found for {}'.format(self))

class ERI(Tensor):

    # Special tensor for the Two-electron integral array. Indices in Physicist's notation

    @staticmethod
    def Antisymmetric(p,q,r,s):

        # Return the antisymmetric tensor ERI as a combination of two regular ERI:
        # <pq||rs> = <pq|rs> - <pq|sr>
        # Note that the subtraction will produce a Collection object as output

        return ERI(p,q,r,s) - ERI(p,q,s,r)

    def __init__(self, p, q, r, s):

        # Four indexes are taken as arguments

        self.idx = [p,q,r,s]
        for index in self.idx:    
            if type(index) != Index:
                raise TypeError('Indexes cannot be {}'.format(type(index)))
        self.create_repr()

    def copy(self):

        # Return a copy of itself

        return ERI(*self.idx)

    def create_repr(self):

        # Create a string representation of ERI

        [p,q,r,s] = self.idx
        self.repr = '<' +  str(p) + str(q) + '|' + str(r) + str(s) + '>'

    def expand(self):

        # If spin-orbital indexes exist, return a Collection with all valid spin cases

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
                                out += ERI(a,b,c,d)
        return out


    def adapt_space(self, verbose = False):

        # Use symmetry properties to put the ERI in a standand form with respect to occupancy space
        # e.g. oovo is transformed into ooov. Spin is not considered.

        standard = [
            'oooo',
            'ooov',
            'oovv',
            'ovov',
            'ovvv',
            'vvvv']

        if verbose: print('\nSpace adapting {}'.format(self))
        space_string = self.space()

        if verbose: print('Current Space String: {}'.format(space_string))

        # If the ERI is alredy in standard form, return a copy
        if space_string in standard:
            if verbose: print('Already in standard form')
            return self.copy()

        # If not, expand in all possible forms and search for the first one that is in standard form 
        else:
            eqforms  = self.equivalent_forms()
            if verbose: print('Checking equivalent forms')
            for X in eqforms:
                if verbose: print('Equivalent form {}. Space string: {}'.format(X, X.space()))
                if X.space() in standard:
                    if verbose: print('Standard form found!')
                    return X.copy()
            raise NameError('No standard form found for {}'.format(self))

    def adapt(self):
    
        # If RHF is true, make all the indexes alpha.

        if self.any_undef_spin():
            raise NameError('Cannot adapt while indexes have undefined spin')

        if self.rhf:
            p,q,r,s = self.idx
            return ERI(p.alpha(), q.alpha(), r.alpha(), s.alpha())
        else:
            return self.copy()

    def equivalent_forms(self):

        # Return a list with all possible permutations of the ERI.

        out = []
        i,j,k,l = self.idx

        out.append(ERI(i,j,k,l))
        out.append(ERI(j,i,l,k))
        out.append(ERI(k,l,i,j))
        out.append(ERI(l,k,j,i))
        out.append(ERI(k,j,i,l))
        out.append(ERI(l,i,j,k))
        out.append(ERI(i,l,k,j))
        out.append(ERI(j,k,l,i))

        return out

class Amplitude(Tensor):

    # Special tensor to represent coupled cluster amplitudes.

    def __init__(self, *argv):
    
        # Takes an even number of indexes as arguments
        # First half of index are taken as occupied indexes (below Fermi), second half as virtual indexes (above Fermi)
    
        self.idx = [] 
        for index in argv:
            if type(index) != Index:
                raise TypeError('Indexes cannot be {}'.format(type(index)))
            self.idx.append(index)

        if len(self.idx) % 2 != 0:
            raise NameError('Amplitudes cannot have odd number of indexes')

        self.rank = int(len(self.idx)/2)
        self.create_repr()


    def copy(self):

        # Return copy of itself

        return Amplitude(*self.idx)

    def create_repr(self):

        # Create string representation of Amplitude

        self.repr = 'T{}('.format(self.rank)
        for n,i in enumerate(self.idx):
            if n == self.rank:
                self.repr += '->'
            self.repr += str(i)
        self.repr += ')'

    def expand(self):

        # If spin-orbital indexes exist, return a Collection with all valid spin cases

        possible = self.spin_cases()
        
        # For amplitude, we require that the number of alphas/betas in the occupied indexes equals the number of alpha/betas in the virtual space
        # e.g. Ij->Ab is allowed, but IJ->ab is not

        # The next few lines uses a numpy trick to iterate through an arbitrary shaped Tensor

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
                    out += Amplitude(*newidx)
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
                out += Amplitude(*newidx).adapt()
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
        return factor*Amplitude(*idx)

class Collection:

    # Object to represent a sum of terms and Contractions

    def __init__(self, terms=[], coef = []):

        self.terms = terms
        self.coef = coef
        self.sort()

    def sort(self):

        # Return a 'sorted' version of the Collection. Sorted in this case has to do with some rules for sorting the terms. Each term is given a 
        # number of points, then they are sorted from smaller to bigger number of points.
        # The points system for each type of tensor is:
        # Fock: -1
        # ERI: -0.5
        # Tn: n
        # General: (# of indexes)/2

        points = []
        for X in self.terms:
            if type(X) == Fock:
                points.append(-1)
            elif type(X) == ERI:
                points.append(-0.5)
            elif type(X) == Amplitude:
                points.append(X.rank)
            elif type(X) == Tensor:
                points.append(len(X.idx)/2)
            elif type(X) == Contraction:
                p = 0
                for Y in X.contracting:
                    if type(Y) == Fock:
                        p += -1
                    elif type(Y) == ERI:
                        p += 1
                    elif type(Y) == Amplitude:
                        p += Y.rank
                    elif type(Y) == Tensor:
                        p += len(Y.idx)/2
                points.append(p)

        s = np.argsort(points)

        self.terms = list(np.array(self.terms)[s])
        self.coef = list(np.array(self.coef)[s])

    # Define addition operations

    def __add__(self, other):

        # If 0 is added, return itself
        if other == 0:
            return Collection(terms=self.terms[:], coef=self.coef[:])

        elif isinstance(other, Tensor) or isinstance(other, Contraction):
            # If other is a tensor or Contraction append it to terms
            newterms = self.terms[:]
            newcoef = self.coef[:]
            newterms.append(other)
            newcoef.append(1)
            return Collection(newterms, newcoef)

        elif type(other) == Collection:
            # For two Contractions, merge the terms
            newterms = self.terms[:] + other.terms[:]
            newcoef = self.coef[:] + other.coef[:]
            return Collection(newterms, newcoef)

        else:
            return TypeError('Addition undefined for collection and {}'.format(type(other)))

    def __radd__(self, other):

        return self + other

    def __sub__(self, other):

        # If 0 is added, return itself
        if other == 0:
            return Collection(terms=self.terms, coef=self.coef)

        elif isinstance(other, Tensor) or isinstance(other, Contraction):
            # If other is a tensor or Contraction append it to
            newterms = self.terms[:]
            newcoef = self.coef[:]
            newterms.append(other)
            newcoef.append(-1)
            return Collection(newterms, newcoef)

        elif isinstance(other, Collection):
            newterms = self.terms[:] + other.terms[:]
            newcoef = self.coef[:]
            for coef in other.coef:
                newcoef.append(-coef)
            return Collection(newterms, newcoef)

    # Iteration thorugh a collection goes through each term

    def __iter__(self):

        return iter(self.terms)

    # Define multiplication

    def __mul__(self, other):

        if isinstance(other, Collection):
            out = Collection()
            for Y,c1 in zip(self.terms, self.coef):
                for X,c2 in zip(other.terms,other.coef):
                    out += (c1*c2)*(Y*X)
            return out

        elif isinstance(other, float) or isinstance(other, int):
            if other == 0:
                return 0
            out = Collection(terms = self.terms[:], coef = self.coef[:])
            out.coef = list(np.array(out.coef)*other)
            return out

        else:
            out = Collection()
            for X,c in zip(self.terms, self.coef):
                new = other*X
                out += c*new
            return out

    def __rmul__(self, other):

        return self*other

    def __pow__(self, other):

        if isinstance(other, Collection):
            out = Collection()
            for Y,c1 in zip(self.terms, self.coef):
                for X,c2 in zip(other.terms, other.coef):
                    out += (c1*c2)*(Y**X)
            return out

        else:
            out = Collection()
            for X,c in zip(self.terms, self.coef):
                out += c*(other**X)
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
        for X,c in zip(self.terms, self.coef):
            if isinstance(X, Tensor):
                out += c*X.expand()
            else:
                out += c*X
        return out

    def adapt(self):
        out = Collection()
        for X,c in zip(self.terms, self.coef):
            out += c*(X.adapt())
        return out

    def adapt_space(self, verbose=False):
        # Just for ERIs and Fock
        out = Collection()
        for X,c in zip(self.terms, self.coef):
            if type(X) in [ERI, Fock, Contraction]:
                out += c*(X.adapt_space(verbose=verbose))
            else:
                out += c*X
        return out

    def simplify(self):
        out = self.expand()
        out = out.adapt()
        l = len(out)

        i = 0
        j = 0
        for i in range(l):
            t1 = out.terms[i]
            c1 = out.coef[i]
            for j in range(i+1,l):
                t2 = out.terms[j]
                c2 = out.coef[j]
                        
                if isinstance(t1, Contraction) and isinstance(t2, Contraction):
                    if t1.isequivalent(t2):
                        out.coef[i] += c2
                        out.coef[j] = 0

                elif t1 == t2:
                    out.coef[i] += c2
                    out.coef[j] = 0
        
        # Clean up zeros
        
        npC = np.array(out.coef)
        npT = np.array(out.terms)

        mask = npC != 0

        out.coef = list(npC[mask])
        out.terms = list(npT[mask])

        return out

class Contraction:

    # Options
    keep_only_connected_terms = False

    @staticmethod
    def contract(*argv):

        # Static contructor function for contrations
        for A in argv:
            if not isinstance(A, Tensor):
                raise TypeError('Contraction not defined for {}'.format(type(A)))
            if A.any_undef_spin():
                raise NameError('Tensor with spin-orbital indexes cannot be contracted: {}'.format(A))

        indexes = []
        for A in argv:
            indexes += A.idx

        internal = []
        external = []
        for i in indexes:
            if np.sum(np.array(indexes) == i) > 1:
                if i not in internal:
                    internal.append(i)
            else:
                external.append(i)

        # If the keep_only_connected_terms option is True:
        # If one of the contracting tensors does not have a index contracting, return 0
        if Contraction.keep_only_connected_terms:
            for A in argv:
                U = np.intersect1d([x for x in A.idx], [x for x in internal])
                if len(U) == 0:
                    return 0

        for i in indexes:
            if i.flip() in indexes:
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

        return Contraction(*argv, inter=internal, ext=external, spin_free=True)

    def __init__(self, *argv, inter, ext, spin_free=False):

        self.int = inter
        self.ext = ext
        self.contracting = list(argv)
        self.sort()
        self.spin_free = spin_free

    def __eq__(self, other):
        
        # Two contractions are equal if each contracting term is the same

        if type(other) != Contraction:
            return False
        
        if len(other.contracting) != len(self.contracting):
            return False

        for A,B in zip(self.contracting, other.contracting):
            if A != B:
                return False 

        return True 

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

        out = ''
        for X in self.contracting:
            out += X.repr + '*'

        return out[:-1]

    def __mul__(self, other):

        if type(other) == float or type(other) == int:
            out = Collection() + self
            return other*out

        if isinstance(other, Tensor):
            if self.spin_free:
                raise NameError('Cannot further contract a Spin-Free contraction (adapted). Try reorder your operations')
            if other.any_undef_spin():
                return other.expand()*self
            C = self.contracting + [other]
            return Contraction.contract(*C)

        if isinstance(other, Collection):
            return Collection.__mul__(other, self)

        else:
            raise TypeError('Multiplication not defined for Contraction and {}'.format(type(other)))

    def __rmul__(self, other):

        return self*other

    def __pow__(self, other):

        # The power operation (**) performs a spin free contraction, that is, a contraction where
        # spin compatibility is not ensured. The terms are simply put together. Use carefully!

        if isinstance(other, Tensor):
            if other.any_undef_spin():
                return other.expand()**self
            C = self.contracting + [other]
            return Contraction.spin_free_contract(*C)
        else:
            raise TypeError('Spin-free contraction not defined for Contraction and {}'.format(type(other)))

    def sub_dummies(self):

        # Currently assuming spin free and RHF !!!!

        label = 1
        Map = {}
        newcontracting = []

        # Return a copy of itself where contracting indexes(dummies) are relabelled. Useful to compare two contractings

        # Loop through each contracting term
        for C in self.contracting:

            # Make a copy of it so we can modify
            newC = C.copy()

            # Loop through indexes of the term
            for i,idx in enumerate(newC.idx):

                # If the index is internal, give it a dummy name
                if idx.alpha() in self.int:
                    # Check if the index has been assigned a dummy index previously (saved in the dictionary)
                    if idx.name in Map:
                        newC.idx[i] = Map[idx.name]
                    # If not, create the assignment and store in the dictionary
                    else:
                        newC.idx[i] = Index(str(label), spin='alpha')
                        Map[idx.name] = Index(str(label), spin='alpha')
                        label += 1
            newcontracting.append(newC)
        
        return Contraction.spin_free_contract(*newcontracting)
        

    def isequivalent(self, other):

        # Test if this contraction is equivalent to another, considering permutation symmetric, dummy indices etc

        # Check type and number of contracting terms
        if type(other) != Contraction:
            return False
        
        contain_ERI = False
        for C1, C2 in zip(self.contracting, other.contracting):
            if type(C1) != type(C2):
                return False
            if type(C1) == ERI:
                contain_ERI = True
                ERI2 = C2

        if len(other.contracting) != len(self.contracting):
            return False

        if not Tensor.rhf:
            raise NameError('UHF case not implemented yet :(')  

        if contain_ERI:
            O = Contraction.spin_free_contract(*other.contracting)
            i = other.contracting.index(ERI2)
    
            for V in ERI2.equivalent_forms():
                O.contracting[i] = V
                if self.sub_dummies() == O.sub_dummies():
                    return True
            return False
        
        else:
            return self.sub_dummies() == other.sub_dummies()
            

    def adapt(self):

        # When adapt is called on a Contraction object each contracting term will be adapted
        # The resulting terms will be put together via a spin free contraction
        # That is, a contraction where spin compatibility  will no be check

        out = Collection()
        out += self.contracting[0].adapt()
        for c in self.contracting[1:]:
            out = out**(c.adapt())
        return out

    def adapt_space(self, verbose=False):

        # When adapt is called on a Contraction object each ERI in the contraction will be space adapted

        out = Collection()
        if type(self.contracting[0]) in [Fock, ERI]:
            out += self.contracting[0].adapt_space(verbose=verbose)
        else:
            out += self.contracting[0]

        if self.spin_free:
            for c in self.contracting[1:]:
                if type(c) == ERI:
                    out = out**(c.adapt_space(verbose=verbose))
                else:
                    out = out**c
        else:
            for c in self.contracting[1:]:
                if type(c) == ERI:
                    out = out*(c.adapt_space(verbose=verbose))
                else:
                    out = out*c
        return out

    def permute(self, x, y):
        
        # Return a copy of itself where indexes x and y were permuted

        newcontracting = self.contracting[:]

        for C in newcontracting:
            for i, idx in enumerate(C.idx):
                if idx == x:
                    C.idx[i] = y
                elif idx == y:
                    C.idx[i] = x

        if self.spin_free:
            return Contraction.spin_free_contract(*newcontracting)
        return Contraction.contract(*newcontracting)
    
