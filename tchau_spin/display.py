from .index import Index
from .tensor import *
from IPython.display import display, Math

def platex(x):

    # Wrapping function to print out the latex representation

    return display(Math(latex_str(x)))

def latex_str(x):

    # Function to create a latex representation of Tensors, Indexes, Collections and Contractions
    if isinstance(x, int) or isinstance(x, float):
        return str(x)

    if isinstance(x, Index):
        # If the spin is not defined, represent with a tilde

        if x.spin == 'any': 
            return r'\tilde{' + x.name + '}'
        else:
            return str(x)

    if isinstance(x, Fock):

        # Add representation for indexes
        out = 'f_{'
        for p in x.idx:
            out += latex_str(p)

        # Close up

        out += '}'
        return out

    if isinstance(x, ERI):

        # Add representation for indexes
        out = r'\langle '
        for p in x.idx[0:2]:
            out += latex_str(p)

        out += '|'

        for p in x.idx[2:]:
            out += latex_str(p)

        out += r'\rangle'
        return out

    if isinstance(x, Amplitude):

        # Add representation for indexes
        out = 't_{'
        for p in x.idx[0:x.rank]:
            out += latex_str(p)

        out += '}^{'

        for p in x.idx[x.rank:]:
            out += latex_str(p)

        out += '}'
        return out

    if isinstance(x, Tensor):

        # Add representation for indexes
        out = x.name + '_{'
        for p in x.idx:
            out += latex_str(p)

        out += '}'
        return out

    if isinstance(x, Contraction):

        # Create string representation of the prefactor

        out = ''
        for C in x.contracting:
            out += latex_str(C)
        
        return out

    if isinstance(x, Collection):

        out = ''
        f = True
        for i,c in zip(x.terms, x.coef):
            if c == 1:
                out += '+'
            elif c == -1:
                out += '-'
            elif c > 0:
                out += '+' + str(c)
            else:
                out += str(c)
            out += latex_str(i)
        return out

    if isinstance(x, list):

        out = '('
        for i in x:
            out += latex_str(i) + ','
        out = out[:-1]
        out += ')'
        return out
