from .index import Index
from .tensor import *
import re
tensor_keys = {
    'A' : ERI.Antisymmetric,
    'V' : ERI,
    'f' : Fock,
    'T' : Amplitude
}

def eqfromlatex(inp):

    # Remove spaces betweem signs and next terms
    out = re.sub('([+-])\s', '\\1', inp)
    # Add * between tensors contraction
    out = re.sub('}\s+?(\w)', '}*\\1', out)
    # Convert the index to my parentesis notation
    out = re.sub('{(\w+?)}_{(\w+?)}', '(\\1,\\2)', out)
    # Convert tensor names
    out = re.sub('f\^', 'f', out)
    out = re.sub('t\^', 'T', out)
    out = re.sub('v\^', 'A', out)
    # Remove fractions
    out = re.sub('\\\\frac{(.+?)}{(\d+?)}', '(1/\\2)*\\1', out)
