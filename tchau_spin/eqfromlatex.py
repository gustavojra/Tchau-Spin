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

def eqfromlatex(inp, index_keys):

    # Generate a Collection object with a latex equation given in the
    # string 'inp' 

    # Remove spaces betweem signs and next terms
    out = re.sub('([+-])\s', '\\1', inp)
    # Add * between tensors contraction
    out = re.sub('}\s+?(\w)', '}*\\1', out)
    # Convert the index to my parentesis notation. Note that the order is inverted, because in my notation hole indices come first, but sympy put it second
    out = re.sub('{(\w+?)}_{(\w+?)}', '(\\2,\\1)', out)
    # Convert tensor names
    out = re.sub('f\^', 'f', out)
    out = re.sub('t\^', 'T', out)
    out = re.sub('v\^', 'A', out)
    # Remove fractions
    out = re.sub('\\\\frac{(.+?)}{(\d+?)}', '(1/\\2)*\\1', out)

    return eqfromstring(out, tensor_keys, index_keys)
