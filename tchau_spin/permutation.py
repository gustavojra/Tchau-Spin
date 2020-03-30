from .tensor import *

class Permutation:

    @staticmethod
    def find_permutations(equation, *perms, verbose=False):

        vprint = print if verbose else lambda *a, **k: None

        if not Tensor.rhf:
            raise NameError('UHF permutations not implemented')

        if not isinstance(equation, Collection):
            raise TypeError('Function argument must be a Collection')

        l = len(equation)
        i = 0
        j = 0

        out = Collection()
        perm_collection = Collection() 

        vprint('\nLooking for permutation of', end= ' ')
        if verbose:
            for p in perms:
                vprint('({},{})'.format(p[0], p[1]), end = ' ')
            print('\n')

        for i in range(l):
            t1 = equation.terms[i]
            c1 = equation.coef[i]
            if not isinstance(t1, Contraction):
                continue
            if c1 == 0:
                continue

            for p in perms:
                t1 = t1.permute(p[0], p[1])

            for j in range(i+1,l):
                t2 = equation.terms[j]
                c2 = equation.coef[j]
                if not isinstance(t2, Contraction):
                    continue
                if c2 == 0:
                    continue
                        
                if t1.isequivalent(t2) and c1 == c2:
                    vprint('\nPermutation pair found: {} -> {}'.format(equation.terms[i], t2))
                    perm_collection += c1*equation.terms[i]
                    equation.coef[i] = 0
                    equation.coef[j] = 0

            vprint('Progress {:<2.1f}%'.format(100*i/l))

        out += equation
        if len(perm_collection) != 0:
            out += Permutation(perms, perm_collection)
        
        # Clean up zeros
        
        vprint('\nCleaning up zeros')
        npC = np.array(out.coef)
        npT = np.array(out.terms)

        mask = npC != 0

        out.coef = list(npC[mask])
        out.terms = list(npT[mask])

        return out

    def __init__(self, permuting_pairs, permuting_eq):

        # Object to represent an equation with a positive permutation
        # e.g. P(ij)W(ij,ab) = W(ij,ab) + W(ji,ab)

        # permuting_pairs must a list of lists of two indices e.g. [[i,j]] or [[i,j], [a,b]]
        # permuting_eq must be a Collection

        self.permuting_pairs = permuting_pairs

        if not isinstance(permuting_eq, Collection):
            raise TypeError('Permutation equation must be an Collection object.')
        self.permuting_eq = permuting_eq

    def __str__(self):

        out = ''
        for p in self.permuting_pairs:
            out += 'P({}{})'.format(*p)

        return out + '{' + str(self.permuting_eq) + '}'

    def __add__(self, other):

        out = Collection() + other

        out.terms.append(self)
        out.coef.append(1)

        return out

    def __radd__(self, other):

        return self + other
