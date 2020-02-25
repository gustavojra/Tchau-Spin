from tensor import Tensor

class fock(Tensor):

    # Object that represents a fock matrix for restricted orbitals

    def __init__(self, index_1, index_2, prefac=1):

        self.idx = [index_1, index_2]
        self.prefac = prefac
        self.repr = 'f(' + str(self.idx[0]) + ',' + str(self.idx[1]) + ')'

    def copy(self):

        return fock(*self.idx, prefac=self.prefac)

    def expand(self):

        possible = self.spin_cases()
        
        # For fock matrix f(p,q) spin of p has to be equal spin q. Thus

        out = collection()
        for p in range(2):
            for q in range(2):
                if possible[p,q] == 1:
                    if p == q:
                        ip = self.idx[0].change_spin(p)
                        iq = self.idx[1].change_spin(q)
                        out += fock(ip, iq, prefac=self.prefac)

        return out
