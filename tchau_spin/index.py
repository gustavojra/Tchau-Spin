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

        return Index(self.name, spin='alpha')

    def beta(self):

        # Return a copy of itself with beta spin

        return Index(self.name, spin='beta')

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

