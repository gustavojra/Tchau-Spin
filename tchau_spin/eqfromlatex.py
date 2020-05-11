from .index import Index
from .tensor import *
from .readeq import eqfromstring
import re

tensor_keys = {
    'A' : ERI.Antisymmetric,
    'V' : ERI,
    'f' : Fock,
    'T' : Amplitude
}

# Given a permutation string (e.g. P(i,j)[X_{i,j}] = X_{i,j} - X_{j,i})

def expand_permutation(string, printv):
    m = re.match(r"P\((\w+?)/(\w+?)\)\[(.+?)\]", string)
    printv('Permutation found: {}'.format(m.group(0)))
    y = ''
    I1 = m.group(1)
    I2 = m.group(2)
    s  = m.group(3)
    y += s
    if s[0] not in '+-':
        s = '+' + s
    
    # Change sign of s
    s = re.sub('[+]', 'PLUS', s)
    s = re.sub('[-]', '+', s)
    s = re.sub('PLUS', '-', s)
    
    for i in I1:
        for j in I2:
            printv('\nPermuting {} and {}'.format(i,j))
            hold = re.sub(i, 'HOLD', s)
            hold = re.sub(j, i, hold)
            hold = re.sub('HOLD', j, hold)
            printv(hold)
            y += ' ' + hold

    return y

def eqfromlatex(inp, index_keys, verbose=False, antisymmetric=True):

    printv = print if verbose else lambda *k,**w: None

    # Generate a Collection object with a latex equation given in the
    # string 'inp' 

    printv('Cleaning up equation...')
    # Remove spaces between signs and tensors
    out = re.sub('([+-])\s', '\\1', inp)
    # Add * between tensors contraction
    out = re.sub('}\s+?(\w)', '}*\\1', out)
    # Convert the index to my parentesis notation. Note that the order is inverted, because in my notation hole indices come first, but sympy put it second
    out = re.sub('{(\w+?)}_{(\w+?)}', '(\\2,\\1)', out)
    # Convert tensor names
    out = re.sub('f\^', 'f', out)
    out = re.sub('t\^', 'T', out)
    if antisymmetric:
        out = re.sub('v\^', 'A', out)
    else:
        out = re.sub('v\^', 'V', out)
    # Remove fractions
    out = re.sub('\\\\frac{(.+?)}{(\d+?)}', '(1/\\2)*\\1', out)

    # Expand permutations
    
    hold = ''
    for term in out.split():
        perms = list(re.findall(r"P\(\w+?/\w+?\)", term))
        if len(perms) == 0:
            hold += term + ' ' 
            continue
        permuting = term
        for p in perms:
            permuting = permuting.replace(p, '')
        
        permuting = permuting.replace('[', '')
        permuting = permuting.replace(']', '')

        for p in perms[::-1]:
            s = p + '[' + permuting + ']'
            expanded = expand_permutation(s, printv)
            permuting = expanded
            printv('\nFinal Expansion on {}:\n'.format(p), expanded)

        hold += permuting + ' '

    out = hold[:-1]


    return eqfromstring(out, tensor_keys, index_keys, verbose=verbose)
