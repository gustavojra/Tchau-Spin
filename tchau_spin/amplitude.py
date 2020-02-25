from tensor import Tensor

class Amplitude(Tensor):

    # Special tensor to represent coupled cluster amplitudes.
    # First half of index are taken as occupied indexes, second half as virtual indexes

    def __init__(self, *argv, prefac=1):

        self.idx = [] 
        for index in argv:
            self.idx.append(index)

        if len(self.idx) % 2 != 0:
            raise NameError('Amplitudes cannot have odd number of indexes')

        self.rank = int(len(self.idx)/2)
        self.prefac = prefac

        # Create string representation
        self.repr = 'T{}('.format(self.rank)
        for n,i in enumerate(self.idx):
            if n == self.rank:
                self.repr += '->'
            self.repr += str(i)
        self.repr += ')'

    def copy(self):

        return Amplitude(*self.idx, prefac=self.prefac)

    def expand(self):

        possible = self.spin_cases()
        
        # For amplitude, we require that the number of alphas/betas in the occupied indexes equals the number of alpha/betas in the virtual space
        # e.g. Ij->Ab is allowed, but IJ->ab is not

        it = np.nditer(possible, flags=['multi_index'])
        out = collection()
        l = self.rank
        while not it.finished:
            if it[0] > 0:
                occ = it.multi_index[0:l]
                vir = it.multi_index[l:]
                if sum(occ) == sum(vir):
                    newidx = []
                    for i in range(len(it.multi_index)):
                        newidx.append(self.idx[i].change_spin(it.multi_index[i]))
                    out += Amplitude(*newidx, prefac=self.prefac)
            it.iternext() 

        return out
