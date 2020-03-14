from .tensor import *

class process_eq:

    # Class to process equation and write out the desired output
    # Translation keys to term equation terms into einsum

    def __init__(self, eq, *ext_ind, name='name', tensor_labels={}):
        
        # To initialize the process, an equation (Collection object)
        # external indexes (i,j,a,b, etc, with spin and space defined)
        # and an optimal name must be given.

        self.name = name
        self.tensor_labels = tensor_labels
        self.eq = eq
        self.external_indexes = ext_ind
        self.ext_string = ''

        for i in self.external_indexes:
            self.ext_string += str(i)

    def get_name(self, X):

        # For a given term, give it a name
        # Standard names start with f, V, and T for fock, ERI and Amplitudes
        # followed by o, O, v or V, depending on the spin and space of the
        # indexes. If the standard name correspond to an entry in the dictionary
        # 'tensor_labels' that entry will be used instead. 

        name_out = ''
        if isinstance(X, Fock):
           name_out += 'f_' 
        elif isinstance(X, ERI):
            name_out += 'V_'
        elif isinstance(X, Amplitude):
            name_out += 'T_'
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

        if name_out in self.tensor_labels:
            return self.tensor_labels[name_out]
        return name_out

    def get_einsum_string(self, X):

        # Produces an einsum string (e.g. 'ijmn,mnab->ijab') from a contraction

        out = '\''

        for C in X.contracting:
            for index in C.idx:
                out += str(index)
            out += ', '

        # Drop the last comma
        out = out[:-2]
        out += ' -> ' + self.ext_string + '\'' 

        return out.lower()

    def einsum_from_contraction(self, X):

        # Produces the einsum command for a contraction

        if isinstance(X, Tensor):
            return self.get_name(X)

        if isinstance(X, Contraction):
            
            out = 'np.einsum({},'.format(self.get_einsum_string(X))
            for C in X.contracting:
                out += ' ' + self.get_name(C) + ','
            out += " optimize = 'optimal')"

            return out
                
    def write_einsums_out(self, output, print_out=False):

        # From a Collection, write all the einsum expressions
        
        out = ''
        
        for element, coef in zip(self.eq.terms, self.eq.coef):
            if coef == 1:
                out += self.name + ' += ' + self.einsum_from_contraction(element) + '\n'
            elif coef == -1:
                out += self.name + ' -= ' + self.einsum_from_contraction(element) + '\n'
            else:
                out += self.name + ' += ' + str(coef) + '*' + self.einsum_from_contraction(element) + '\n'

        with open(output, 'w') as outp:
            if print_out:
                print(out)
            outp.write(out)
