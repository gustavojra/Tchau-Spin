class Index:

    # Class to represente indexes using in quantum chemistry calculations
    # Must have a name (e.g. i,j,k,l,m,n, etc)
    # Msut have a spin (alpha, beta or any)

    @staticmethod
    def new(names, spins = '', spaces = ''):

        # Auxiliar function to create several indexes at the same time
        # e.g. new('ijab', 'abab', 'hhpp') creates I (hole), j (hole), A (particle)
        # and b (particle). spins and spaces are optinal. Use 'x' for any.
        # e.g. new('ij', 'ax') creates I and ~j (spin orbital)

        while len(spins) < len(names):
            spins = spins + 'x'

        while len(spaces) < len(names):
            spaces = spaces + 'x'

        holes = []
        par = []
        for k in spaces:
            if k == 'h':
                holes.append(True)
                par.append(False)
            elif k == 'p':
                holes.append(False)
                par.append(True)
            else:
                holes.append(False)
                par.append(False)

        out = []
        for i in range(len(names)):
            if spins[i] == 'a':
                new_index = Index(names[i], spin = 'alpha', hole = holes[i], particle=par[i])
            elif spins[i] == 'b':
                new_index = Index(names[i], spin = 'beta', hole = holes[i], particle=par[i])
            else:
                new_index = Index(names[i], hole = holes[i], particle=par[i])
            
            out.append(new_index)

        return out
    
    def __init__(self, name, spin='any', hole=False, particle=False):

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

        self.hole = hole
        self.particle = particle

        if self.hole and self.particle:
            raise NameError('Index cannot be particle and hole at the same time')

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
            return ((self.name, self.spin) == (other.name, other.spin))
        return False

    def __ne__(self,other):

        if type(other) == Index:
            return ((self.name, self.spin) != (other.name, other.spin))
        return False

    def __lt__(self,other):

        if type(other) == Index:
            return self.name < other.name
        return False

    def __le__(self,other):

        if type(other) == Index:
            return self.name <= other.name
        return False

    def __gt__(self,other):

        if type(other) == Index:
            return self.name > other.name
        return False

    def __ge__(self,other):

        if type(other) == Index:
            return self.name >= other.name
        return False

    def alpha(self):

        # Return a copy of itself with alpha spin

        return Index(self.name, spin='alpha', hole=self.hole, particle=self.particle)

    def beta(self):

        # Return a copy of itself with beta spin

        return Index(self.name, spin='beta', hole=self.hole, particle=self.particle)

    def change_spin(self, s):

        # Return a copy of the index with the desired spin (0 for beta, 1 for alpha)

        if s == 1:
            return self.alpha()

        elif s == 0:
            return self.beta()

        else:
            raise ValueError('Invalid spin input. Must be 0 (for beta) or 1 (for alpha)')

    def flip(self):

        # Return copy with opposite spin

        if self.spin == 'alpha':
            return self.beta()
        elif self.spin == 'beta':
            return self.alpha()
        else:
            raise NameError('Cannot flip undefined spin')

