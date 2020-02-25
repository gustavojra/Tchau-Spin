from tensor import Tensor

class contraction:

    @staticmethod
    def contract(A,B):

        # Static contructor function for contrations

        if isinstance(A, Tensor) and isinstance(B, Tensor):
            if A.any_undef_spin() or B.any_undef_spin():
                raise NameError('Tensor with undefined spin cannot be contracted')

            internal = []
            external = []
            for i in A.idx + B.idx:
                if i in A.idx and i in B.idx:
                    if i not in internal:
                        internal.append(i)
                else:
                    external.append(i)
            if len(internal) == 0:
                return 0
            for i in external:
                if i.flip() in external:
                    return 0
            else:
                return contraction(A, B, inter=internal, ext=external, prefac=A.prefac*B.prefac)
            
        else:
            raise TypeError('Contraction not defined for {} and {}'.format(type(A), type(B)))

    def __init__(self, *argv, inter, ext, prefac=1):

        self.int = inter
        self.ext = ext
        self.prefac = prefac
        self.contracting = list(argv)

    def __str__(self):

        out = str(self.prefac)
        for X in self.contracting:
            out += '*' + X.repr

        return out

    def __mul__(self, other):

        if type(other) == float or type(other) == int:
            return contraction(*self.contracting, inter=self.int, ext=self.ext, prefac=self.prefac*other)

    def __rmul__(self, other):

        return self*other
