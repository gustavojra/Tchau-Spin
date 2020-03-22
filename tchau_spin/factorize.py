from .index import Index
from .tensor import *

class Factor:

    # Object that represents a factorized expression (A+B)*(C+D)

    @staticmethod
    def factorize_ERI(eq, verbose=False):

        # Takes an equation (Collection) as input and returns another equation (Collection) where ERI tensors are factored out

        if not isinstance(eq, Collection):

            # Check if the input is a collection

            raise TypeError('Only Collection objects can be factorized')

        out = Collection()
        ignore = []

        # Loop through the terms
        for i, X in enumerate(eq.terms):

            # If index is in ignore it means that it has already been accounted for or it is not valid for factorization
            if i in ignore:
                continue
            print('\nTrying to factorize {}'.format(X))

            # Check if it is a Contraction
            # If not, append the object to the output with the coefficient
            if not isinstance(X, Contraction):
                out += eq.coef[i]*X
                print('Not a contraction')
                continue

            V1 = X.get_ERI()
            if V1 == None:
                # If the contraction does not have a ERI, just append it to the output
                out += eq.coef[i]*X
                print('No ERI')
                continue

            first_match = True
            # Loop through the terms again with j > i
            for j, Y in enumerate(eq.terms[i+1:]):
                if i+j+1 in ignore:
                    continue

                print('Comparing with {}'.format(Y))

                # Check if it is a Contraction
                # If not, append the object to the output with the coefficient
                if not isinstance(Y, Contraction):
                    out += eq.coef[i+j+1]*Y
                    ignore.append(i+j+1)
                    print('Not a contraction')
                    continue

                V2 = Y.get_ERI()
                if V2 == None:
                    # If the contraction does not have a ERI, just append it to the output
                    out += eq.coef[i+j+1]*Y
                    ignore.append(i+j+1)
                    print('No ERI')
                    continue

                # Test if the ERI within the two contractions is the same
                if V1 in V2.equivalent_forms():
                    print(i+j+1)
                    ignore.append(i+j+1)
                    print('First Match!')
                    c2 = Y.remove(V2)

                    # If this is the first match, create a new factor
                    if first_match:
                        c1 = X.remove(V1)
                        newfactor =  Factor(V1, eq.coef[i]*c1 + eq.coef[i+j+1]*c2)
                        first_match = False

                    # If this is another match, append the new terms to the existing factor
                    else:
                        print('New Match!')
                        newfactor.c2 += eq.coef[i+j+1]*c2

            # If no match has been found, just append the contraction back into the output
            if first_match:
                print('No match')
                out += eq.coef[i]*X
            else:
                out +=  newfactor


        return out


    def __init__(self, c1, c2):

        self.c1 = 1*c1
        self.c2 = 1*c2

    def __str__(self):

        # Return a string representation of the Factor
        return '[' + str(self.c1) + ']*[' + str(self.c2) + ']'

    def distribute(self):

        # Expand out the factorization back to a Collection of Contractions

        out = Collection()
        for x in self.c1:
            for y in self.c2:
                out += x*y

        return out

    def __add__(self, other):

        out = Collection() + other

        out.terms.append(self)
        out.coef.append(1)

        return out

    def __radd__(self, other):

        return self + other

    def __mul__(self, other):

        if type(other) in [int, float]:
            
            return 3*(Collection() + self)

        else:

            raise TypeError('Factor object do not support multiplication by non-numerical objects')

    def __rmul__(self, other):

        if type(other) in [int, float]:
            
            return self*other

        else:

            raise TypeError('Factor object do not support multiplication by non-numerical objects')
