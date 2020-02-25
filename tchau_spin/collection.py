
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

    def __sub__(self, other):

        # If 0 is added, return itself
        if other == 0:
            return collection(terms=self.terms)

        elif isinstance(other, Tensor) or isinstance(other, contraction):
            # If other is a tensor or contraction append it to
            newother = other.copy()
            newother.prefac *= -1
            newterms = self.terms[:]
            newterms.append(newother)
            return collection(newterms)

        elif isinstance(other, collection):
            newterms = other.terms[:]
            for X in newterms:
                X.prefac *= -1
            newterms = self.terms[:] + newterms
            return collection(newterms)

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

    def expand(self):
        out = collection()
        for X in self:
            if isinstance(X, Tensor):
                out += X.expand()
            else:
                out += X
        return out
