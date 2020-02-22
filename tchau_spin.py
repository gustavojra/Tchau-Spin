import numpy as np

# Index objects

class Index:
    
    def __init__(self, name, spin='any'):

        if len(name) > 1:
            raise NameError('Index must be identified by one letter')
        self.name == name
        self.spin == spin
        if spin == 'alpha':
            self.s = 1
        elif spin == 'beta':
            self.s = 0
        elif spin =='any':
            self.s = 2
        else:
            raise NameError('Invalid Spin Type. Must be alpha, beta or any')

    def __str__(self)

        if self.s == 1:
            return self.name.upper()
        elif self.s == 0:
            return self.name.lower()
        else:
            return '~'+self.name

    def flip(self)

        # Flip the spin

        if self.s == 1:

            return Index(name=name, spin = 'alpha')

        if self.s == 1:

            return Index(name=name, spin = 'beta')

        else:
            raise NameError('Cannot flip undefined spin')


class collection:

    # Collection of terms

    def __init__(self):

        self.terms = []

    def __add__(self, other):

        self.terms = []

    def __str__(self, other):

        out  = ''
        for x in terms:
            out += str(x)

        print(out)


class fock_rhf:

    # Object that represents a fock matrix for restricted orbitals

    def __init__(self, index_1, index_2):

        self.idx = [index_1, index_2]
        self.prefac = 1.0

    def __str__(self):

        if self.prefac != 1:

            return str(self.prefac) +'*f(' + str(self.i1) + ',' + str(self.i2) + ')'

        return 'f(' + str(self.i1) + ',' + str(self.i2) + ')'

    def spin_cases(self):

        possible = np.ones((2,2))
        
        # For fock matrix f(p,q) spin of p has to be equal spin q. Thus

        for p in range(2):
            for q in range(2):
                if p != q:
                    possible[p,q] = 0

        # Check if any index has spin defined:
        if self.i1 == 0:
            possible[1,:] = 0

        if self.i1 == 1:
            possible[0,:] = 0

        out = collection()

            

    def spin_simplify(self):

        if any([x.s == 2 for x in self.idx]):
            
            raise NameError('Cannot simplify with spin orbital indexes')

        else:

            pass

                
class Amplitude:

    def __init__(self, name, 



