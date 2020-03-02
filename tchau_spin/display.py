from .index import Index
from .tensor import *
from IPython.display import display, Math

def platex(x):

    # Wrapping function to print out the latex representation

    return display(Math(latex_str(x)))

def latex_str(x):

    # Function to create a latex representation of Tensors, Indexes, Collections and Contractions

    if isinstance(x, Index):
        # If the spin is not defined, represent with a tilde

        if x.spin == 'any': 
            return r'\tilde{' + x.name + '}'
        else:
            return str(x)

    if isinstance(x, Fock):

        # Create string representation of the prefactor

        if x.prefac == 1:
            pf = ''
        elif x.prefac == -1:
            pf = '-'
        else:
            pf = str(x.prefac)

        # Add representation for indexes
        out = pf + 'f_{'
        for p in x.idx:
            out += latex_str(p)

        # Close up

        out += '}'
        return out

    if isinstance(x, ERI):

        # Create string representation of the prefactor

        if x.prefac == 1:
            pf = ''
        elif x.prefac == -1:
            pf = '-'
        else:
            pf = str(x.prefac)

        # Add representation for indexes
        out = pf + r'\langle '
        for p in x.idx[0:2]:
            out += latex_str(p)

        out += '|'

        for p in x.idx[2:]:
            out += latex_str(p)

        out += r'\rangle'
        return out

    if isinstance(x, Amplitude):

        # Create string representation of the prefactor

        if x.prefac == 1:
            pf = ''
        elif x.prefac == -1:
            pf = '-'
        else:
            pf = str(x.prefac)

        # Add representation for indexes
        out = pf + 't_{'
        for p in x.idx[0:x.rank]:
            out += latex_str(p)

        out += '}^{'

        for p in x.idx[x.rank:]:
            out += latex_str(p)

        out += '}'
        return out

    if isinstance(x, Tensor):

        # Create string representation of the prefactor

        if x.prefac == 1:
            pf = ''
        elif x.prefac == -1:
            pf = '-'
        else:
            pf = str(x.prefac)

        # Add representation for indexes
        out = pf + x.name + '_{'
        for p in x.idx:
            out += latex_str(p)

        out += '}'
        return out

    if isinstance(x, Contraction):

        # Create string representation of the prefactor

        if x.prefac == 1:
            pf = ''
        elif x.prefac == -1:
            pf = '-'
        else:
            pf = str(x.prefac)

        out = str(pf) 
        for C in x.contracting:
            out += latex_str(C)
        
        return out

    if isinstance(x, Collection):

        out = ''
        f = True
        for item in x:
            if item.prefac > 0:
                out += '+'
            out += latex_str(item)
        return out
