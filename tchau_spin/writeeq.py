from .tensor import *
from .factorize import Factor

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
            name_out += X.name + '_'

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

    def get_einsum_string(self, X, target):

        # Produces an einsum string (e.g. 'ijmn,mnab->ijab') from a contraction

        out = '\''

        for C in X.contracting:
            for index in C.idx:
                out += str(index)
            out += ', '

        # Drop the last comma
        out = out[:-2]
        out += ' -> ' + target + '\'' 

        return out.lower()

    def einsum_from_contraction(self, X, target):

        # Produces the einsum command for a contraction

        if isinstance(X, Tensor):
            return self.get_name(X)

        if isinstance(X, Contraction):
            
            out = 'np.einsum({},'.format(self.get_einsum_string(X, target))
            for C in X.contracting:
                out += ' ' + self.get_name(C) + ','
            out += " optimize = 'optimal')"

            return out

    def find_external_indexes(self, X):

        # From a Collection, find external indexes
        if isinstance(X.terms[0], Tensor):
            out = ''
            for i in X.terms[0].idx:
                out += i.name

        elif isinstance(X.terms[0], Contraction):
            out = ''
            for i in X.terms[0].ext:
                out += i.name

        return out

    def intermediate_from_collection(self, X, name):

        # Determine 'local' external indexes from the first element of the Collection
        local_idx = self.find_external_indexes(X)
        if isinstance(X.terms[0], Tensor):
            out = name + ' = ' + str(X.coef[0]) + '*copy.deepcopy(' + self.get_name(X.terms[0]) + ')\n'

        elif isinstance(X.terms[0], Contraction):
            out = name + ' = ' + str(X.coef[0]) + '*' + self.einsum_from_contraction(X.terms[0], local_idx) + '\n'

        # The following terms will be transpose to be added to the first
        for t, c in zip(X.terms[1:], X.coef[1:]):
            if isinstance(t, Tensor):
                idx_names = [i.name for i in t.idx]

                # Create a tranpose string to performed the appropriated transposition before addition
                count = 0
                transpose_string = '('
                ordered = '('        
                for target_i in local_idx:
                    ordered += str(count) + ','
                    count += 1
                    try:
                        transpose_string += str(idx_names.index(target_i)) + ','
                    except ValueError:
                        raise ValueError('Error adding {} and {} together. Check indexes'.format(X.terms[0], c))

                if transpose_string == ordered:
                    out += name + ' += ' + str(c) + '*' + self.get_name(t) + '\n'
                    continue

                else:
                    transpose_string = transpose_string[:-1] + ')'
                    out += name + ' += ' + str(c) + '*' + self.get_name(t) + '.transpose' + transpose_string + '\n'

            elif isinstance(t, Contraction):
                out += name + ' += ' + str(c) + '*' + self.einsum_from_contraction(t, local_idx) + '\n'

        return out
            
                
    def write_einsums_out(self, output='', print_out=False):

        # From a Collection, write all the einsum expressions
        
        out = ''
        
        for element, coef in zip(self.eq.terms, self.eq.coef):

            if not isinstance(element, Factor):
                if coef == 1:
                    out += self.name + ' += ' + self.einsum_from_contraction(element, self.ext_string) + '\n'
                elif coef == -1:
                    out += self.name + ' -= ' + self.einsum_from_contraction(element, self.ext_string) + '\n'
                else:
                    out += self.name + ' += ' + str(coef) + '*' + self.einsum_from_contraction(element, self.ext_string) + '\n'
            else:

                
                if len(element.c1) == 1:
                    A = element.c1.terms[0]
                else:
                    A = Tensor('A', *elements.c2.terms[0].idx)
                    out += self.intermediate_from_collection(element.c1, self.get_name(A))

                B = Tensor('B', *element.c2.terms[0].idx)
                out += self.intermediate_from_collection(element.c2, self.get_name(B))
                out += self.name + ' += ' + self.einsum_from_contraction(A**B, self.ext_string) + '\n'

        # If no output was given, nothing will be saved to disk
        if output != '':
            with open(output, 'w') as outp:
                outp.write(out)

        # If print_out was set to True, print to screen
        if print_out:
            print(out)

        else:
            return out
