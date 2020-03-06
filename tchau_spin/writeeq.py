from .tensor import *

class process_eq:

    # Class to process equation and write out the desired output

    # Translation keys to term equation terms into einsum
    translation = {}

    def __init__(self, eq, name='name'):
        
        self.eq = eq

    def get_name(self, X):

        name_out = ''
        if isinstance(X, Fock):
           name_out += 'f' 
        elif isinstance(X, ERI):
            name_out += 'V'
        elif isinstance(X, Amplitude):
            name_out += 'T'
        else:
            name_out += X.name

        for i in X.idx:
            if i.spin == 'alpha':
                if i.hole:
                    name_out += 'O'
                elif i.particle:
                    name_out += 'V'
            elif i.spin == 'beta':
                if i.hole:
                    name_out += 'o'
                elif i.particle:
                    name_out += 'v'

        return name_out

    def get_einsum_string(self, X):

        out = ''

        for C in X.contracting:
            for index in C.idx:
                out += str(index)

                # Figure out output string eg ijab (PUT IN THE INPUT?)
                # CONTINUE HERE
                

    def einsum_from_contraction(self, X):

        if isinstance(X, Tensor):
            return self.get_name(X)

        if isinstance(X, Contraction):
            
            out = 'np.einsum({}, {})'  

    def write_einsums_out(self, output, print_out=False):
        
        out = ''
        
        for elementi, coef in zip(self.eq.terms, self.eq.coef):
            out += 'name += ' + coef + '*' + self.einsum_from_contraction(element)

        with open(output, 'w') as outp:
            if print_out:
                print(out)
            outp.write(out)
