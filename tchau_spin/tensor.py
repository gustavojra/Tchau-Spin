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

    def __sub__(self, other):

        c = collection() + self
        
        return c - other

    def __rsub__(self, other):

        c = collection() - self

        return c + other

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
