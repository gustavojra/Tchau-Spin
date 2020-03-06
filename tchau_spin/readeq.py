from .index import Index
from .tensor import *
import re

def read_equation(inp):

    # Read an equation from an input file

    with open(inp) as eqin:
        
        eqout = Collection()
        f_options = False
        f_eq = False
        idx_keys = {}
        tensor_keys = {
            'A' : ERI.Antisymmetric,
            'V' : ERI,
            'f' : Fock,
            'T' : Amplitude
        }

        for line in eqin:

            if line[0] == '#':
                continue
            
            if 'OPTIONS' in line:
                f_options = True
                continue

            if 'END' in line:
                f_options = False
                f_eq = False
                continue

            if 'EQUATION' in line:
                f_options = False
                f_eq = True
                continue

            # Read options

            if f_options:
                try:
                    # Read indexes
                    ind_info = line.split()

                    # First entry must be the index symbol, max one letter
                    if len(ind_info[0]) > 1:
                        raise NameError('Index symbol can only be one letter') 

                    index_name = ind_info[0]
                    if len(ind_info) == 1:
                        idx_keys[index_name] = Index(index_name)

                    if len(ind_info) == 2:
                        idx_keys[index_name] = Index(index_name, spin=ind_info[1])

                    if len(ind_info) == 3:
                        if ind_info[2] == 'hole':
                            idx_keys[index_name] = Index(index_name, spin=ind_info[1], hole=True)
                        elif ind_info[2] == 'particle' or ind_info[2] == 'par':
                            idx_keys[index_name] = Index(index_name, spin=ind_info[1], particle=True)
                        else:
                            raise NameError('Invalid space type {}'.format(ind_info[2]))

                except ValueError:
                    raise NameError('Invalid option line: {}'.format(line))

            # Read equation

            if f_eq:

                # Slice the string with respect to spaces.
                # Space is interpreted as the end of a terms and beginning of another

                terms = line.split() 

                # Phase is created to keep track of any loose +,- signs
                phase = 1

                # Loop through terms
                for t in terms:

                    # If terms is just a positive sign, make phase = +1
                    if t == '+':
                        phase = +1
                        continue

                    # If terms is just a negative sign, make phase = -1
                    elif t == '-':
                        phase = -1
                        continue

                    # If no * is found in the term, it is a stand alone tensor
                    if not '*' in t: 
                         # Test if there is a sign in front of the term
                         if t[0] == '-':
                             phase = -1
                         elif t[0] == '+':
                             phase = 1
                         
                         # Call a function to create a tensor for this term
                         new = create_tensor(t, tensor_keys, idx_keys)

                         # Add it to the main output
                         eqout += phase*new

                    # if * is in the term it means that it is a contraction
                    if '*' in t:

                        sub = Collection()
                        subterms = t.split('*')

                        # Thus, we loop through each (sub)terms of the contraction
                        for st in subterms:
                            try:
                                # Test if the term is just a number
                                phase = float(st)*phase
                                continue
                            except ValueError:
                                # If not, test if the first character is a positve or negative sign, saving it in the phase
                                if st[0] == '-':
                                    phase = -1
                                elif st[0] == '+':
                                    phase = 1

                                # Call a function to process the individual term, creating a tensor object
                                new = create_tensor(st, tensor_keys, idx_keys)

                                # If the sub collection is empty, add to it
                                if len(sub) == 0:
                                    sub += new

                                # If not, contract what is there with the new term
                                else:
                                    sub = sub*new 

                        # Once we done collecting the terms of the contraction, add it to the main output
                        eqout += phase*sub

                
    return eqout
            
                
def create_tensor(string, tensor_keys, index_keys):
    
    # Return a tensor from a string

    m = re.search('(\S)\((\S+?)\)', string)

    if not m:
        raise NameError('Equation component not reconized: {}'.format(string))

    indexes = []
    for k in m.group(2).replace(',', '') :
        try:
            indexes.append(index_keys[k])
        except KeyError:
            raise KeyError('Index {} not defined previously.'.format(k))
    
    try:
        new = tensor_keys[m.group(1)](*indexes)
    except KeyError:
        new = Tensor(name=m.group(1), *indexes)

    return new
