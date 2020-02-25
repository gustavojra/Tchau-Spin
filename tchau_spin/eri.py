from tensor import Tensor

class ERI(Tensor):

    # Special tensor for the Two-electron integral array. Indices in Physicist's notation

    @staticmethod
    def Antisymmetric(p,q,r,s, prefac=1):
        return ERI(p,q,r,s) - ERI(p,q,s,r)

    def __init__(self, p, q, r, s, prefac=1):

        self.idx = [p,q,r,s]
        self.prefac = prefac
        self.repr = '<' +  str(p) + str(q) + '|' + str(r) + str(s) + '>'

    def copy(self):

        return ERI(*self.idx, prefac=self.prefac)

    def expand(self):

        possible = self.spin_cases()
        
        # For the ERI <pq|rs> we have the spin condition p=r and q=s

        out = collection()
        for p in range(2):
            for q in range(2):
                for r in range(2):
                    for s in range(2):
                        if possible[p,q,r,s] == 1:
                            if p == r and q == s:
                                a = self.idx[0].change_spin(p)
                                b = self.idx[1].change_spin(q)
                                c = self.idx[2].change_spin(r)
                                d = self.idx[3].change_spin(s)
                                out += ERI(a,b,c,d, prefac=self.prefac)
        return out
