from .tensor import *
from .factorize import Factor
from .permutation import Permutation
import re

class eq_to_julia:

    # Class to process equation and write out the desired output
    # Translation keys to term equation terms into einsum

    def __init__(self, eq, *ext_ind, name='name', tensor_labels={}):
        
        # To initialize the process, an equation (Collection object)
        # external indexes (i,j,a,b, etc, with spin and space defined)
        # and an optimal name must be given.

        self.name = name
        self.final_name = name
        self.tensor_labels = tensor_labels
        self.eq = eq
        self.external_indexes = ext_ind
        self.ext_string = ''

        if len(ext_ind) > 0:
            self.final_name += '['
            for i in self.external_indexes:
                self.ext_string += str(i) + ','
                self.final_name += str(i).lower() + ','
            self.ext_string = self.ext_string[:-1]
            self.final_name = self.final_name[:-1] + ']'

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

    def get_tensor_string(self, X):
        
        # From a tensor Xijab get a string with its index e.g. i,j,a,b

        out = ''
        for i in X.idx:
            out += str(i) + ','

        out = out[:-1]
        return self.get_name(X) + '[' + out.lower() + ']'

    def get_contraction_string(self, X):

        # From a contraction get a string

        if isinstance(X, Tensor):
            return self.get_tensor_string(X)

        out = ''
        for C in X.contracting:
            out += self.get_tensor_string(C) + '*'

        out = out[:-1]

        return out

    def intermediate_from_collection(self, X, name):

        out = name + ' := ' + str(X.coef[0]) + '*' +  self.get_contraction_string(X.terms[0]) + '\n'

        for t, c in zip(X.terms[1:], X.coef[1:]):
            out += name + ' += ' + str(c) + '*' + self.get_contraction_string(t) + '\n'

        return out

    def loop_index_out(self, inp, pulling):

        # Input must be a list of TensorOperations commands

        # Dictionary with Tensors that slices were created already
        viewd = dict()

        # Output
        out = []

        # Loop through commands (lines in the input)
        for comm in inp:

            if comm == '':
                continue

            # Find tensors in the line (e.g. T2[i,j,a,b])
            tensors = re.findall(r"(\w+?\[.+?\])", comm)

            # Store new command
            new_comm = comm

            # Loop through tensors found above.
            for t in tensors:
                # Save the tensor by itself (e.g. T2[i,j,a,b]  -> T2)
                t_solo = re.sub(r'\[([\w,]+?)\]', '', t)

                # Save indexes (e.g. T2[i,j,a,b] -> ['i', 'j', 'a', 'b'])
                indexes = re.search(r'\[([\w,]+?)\]', t).group(1).split(',')

                # List to store indexes that will be substituted (those given in 'pulling')
                subs = []

                # idx_braket will store the new indexes (e.g. if you are pulling i,j out [i,j,a,b] becomes
                #  [a,b])
                idx_braket = '['
                for i,ind in enumerate(indexes):
                    if ind in pulling:
                        # If a index from pulling is found in the tensor, save into subs list
                        subs.append((i,ind))
                    else:
                        idx_braket += ind + ','

                # If subs list has nothing, it means nothing has to be done to this tensor
                if len(subs) == 0:
                    continue

                # Start creating a new name for the tensor. If we pull i and j out
                # T2[i,j,a,b] will be called T2_1i2j. A slice (view in julia), will be created with this name
                newt = t_solo + '_'
                for i,ind in subs:
                    newt += str(i+1) + ind

                # If the new tensor name is save in viewd, it has been created already. If not, create it
                if newt not in viewd:
                    create_view = newt + " = view({},".format(t_solo)
                    for ind in indexes:
                        if ind in pulling:
                            create_view += ind + ','
                        else:
                            create_view += ':,'
                    viewd[newt] = create_view[:-1] + ')'

                newt += idx_braket[:-1] + ']'

                # Replace the command with the new syntax
                new_comm = new_comm.replace(t, newt)

            # Append the new command to the list
            out.append(new_comm)

        out = [viewd[k] for k in viewd] + ['\n'] + out

        return out

    def write_tensorop_out(self, output='', print_out=False, pull_index_out = []):

        # From the processed collection, write TensorOperations syntax for Julia

        out = ''

        for element, coef in zip(self.eq.terms, self.eq.coef):

            # If the term is a factor
            if isinstance(element, Factor):
                
                if len(element.c1) == 1:
                    A = element.c1.terms[0]
                else:
                    # Create a intermediate array to hold the result factorized (left) terms
                    A = Tensor('A', *elements.c2.terms[0].idx)
                    out += self.intermediate_from_collection(element.c1, self.get_tensor_string(A))

                # Create a intermediate array to hold the result factorized (right) terms
                B = Tensor('B', *element.c2.terms[0].idx)
                out += self.intermediate_from_collection(element.c2, self.get_tensor_string(B))
                out += self.final_name + ' += ' + self.get_contraction_string(A**B) + '\n'

            elif isinstance(element, Permutation):

                # Create a intermediate array to hold the result before the permutation

                # Save indexes of the first elements (which will represent the external indexes of the whole collection)
                P = Tensor('P', *self.external_indexes)
                out += self.intermediate_from_collection(element.permuting_eq, self.get_tensor_string(P)) + '\n'

                transpose = [i.name for i in self.external_indexes]
                print('tranpose strin: {}'.format(transpose))
                idx_string = self.ext_string.lower().replace(',','')
                print('idx_string: {}'.format(idx_string))
                for p in element.permuting_pairs:
                    print(p)

                    p1 = idx_string.index(p[0].name)
                    p2 = idx_string.index(p[1].name)

                    print(p1, p2)
                    transpose[p1], transpose[p2] = transpose[p2], transpose[p1]

                perm = self.get_name(P) + '[' 
                for p in transpose:
                    perm += p + ','
                perm = perm[:-1] + ']'

                out += self.final_name + ' += ' + self.get_tensor_string(P) + ' + ' + perm + '\n'

            # Lastly, if the term is a Contraction or Tensor
            else:
                if coef == 1:
                    out += self.final_name + ' += ' + self.get_contraction_string(element) + '\n'
                elif coef == -1:
                    out += self.final_name + ' -= ' + self.get_contraction_string(element) + '\n'
                else:
                    out += self.final_name + ' += ' + str(coef) + '*' + self.get_contraction_string(element) + '\n'
        
        # Pull out the desired indexes (to be used as loops)

        if len(pull_index_out) > 0:
            inp = out.split('\n')
            out = '\n'.join(self.loop_index_out(inp, pull_index_out))

        # If no output was given, nothing will be saved to disk
        if output != '':
            with open(output, 'w') as outp:
                outp.write(out)

        # If print_out was set to True, print to screen
        if print_out:
            print(out)
            return out

        else:
            return out
