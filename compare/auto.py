import psi4
import os
import sys
import numpy as np
import time
import copy

np.set_printoptions(suppress=True, linewidth=120)

class RCCSD:

    def update_energy(self):
        
        # Energy equation
        self.Ecc = 2.0*np.einsum('kc, kc -> ', self.fock_OV, self.T1, optimize = 'optimal')
        self.Ecc += 2.0*np.einsum('kldc, lkcd -> ', self.T2, self.Voovv, optimize = 'optimal')
        self.Ecc -= np.einsum('lkcd, klcd -> ', self.T2, self.Voovv, optimize = 'optimal')
        self.Ecc += 2.0*np.einsum('lc, kd, lkcd -> ', self.T1, self.T1, self.Voovv, optimize = 'optimal')
        self.Ecc -= np.einsum('lc, kd, klcd -> ', self.T1, self.T1, self.Voovv, optimize = 'optimal')

    def update_amp(self):

        # Create a new set of amplitudes
        newT1 = np.zeros(self.T1.shape)
        newT2 = np.zeros(self.T2.shape)

        # T1 Amplitude equations
        newT1 += self.fock_OV
        newT1 -= np.einsum('ik, ka -> ia', self.fock_OO, self.T1, optimize = 'optimal')
        newT1 += np.einsum('ca, ic -> ia', self.fock_VV, self.T1, optimize = 'optimal')
        newT1 -= np.einsum('kc, ic, ka -> ia', self.fock_OV, self.T1, self.T1, optimize = 'optimal')
        newT1 += 2.0*np.einsum('kc, ikac -> ia', self.fock_OV, self.T2, optimize = 'optimal')
        newT1 -= np.einsum('kc, kiac -> ia', self.fock_OV, self.T2, optimize = 'optimal')
        newT1 += 2.0*np.einsum('kc, kica -> ia', self.T1, self.Voovv, optimize = 'optimal')
        newT1 -= np.einsum('kc, icka -> ia', self.T1, self.Vovov, optimize = 'optimal')
        newT1 += 1.5*np.einsum('ikcd, kadc -> ia', self.T2, self.Vovvv, optimize = 'optimal')
        newT1 += -2.0*np.einsum('kc, la, lkic -> ia', self.T1, self.T1, self.Vooov, optimize = 'optimal')
        newT1 += 0.5*np.einsum('kicd, kacd -> ia', self.T2, self.Vovvv, optimize = 'optimal')
        newT1 += 2.0*np.einsum('kc, id, kacd -> ia', self.T1, self.T1, self.Vovvv, optimize = 'optimal')
        newT1 += -2.0*np.einsum('lkac, lkic -> ia', self.T2, self.Vooov, optimize = 'optimal')
        newT1 += -0.5*np.einsum('ikcd, kacd -> ia', self.T2, self.Vovvv, optimize = 'optimal')
        newT1 += -0.5*np.einsum('kicd, kadc -> ia', self.T2, self.Vovvv, optimize = 'optimal')
        newT1 -= np.einsum('kc, id, kadc -> ia', self.T1, self.T1, self.Vovvv, optimize = 'optimal')
        newT1 += np.einsum('kc, la, klic -> ia', self.T1, self.T1, self.Vooov, optimize = 'optimal')
        newT1 += np.einsum('lkac, klic -> ia', self.T2, self.Vooov, optimize = 'optimal')
        newT1 += np.einsum('ic, lkad, klcd -> ia', self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT1 += 4.0*np.einsum('kc, ilad, klcd -> ia', self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT1 += -2.0*np.einsum('kc, id, la, klcd -> ia', self.T1, self.T1, self.T1, self.Voovv, optimize = 'optimal')
        newT1 += -1.5*np.einsum('la, ikcd, lkcd -> ia', self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT1 += -2.0*np.einsum('kc, liad, klcd -> ia', self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT1 += -2.0*np.einsum('ic, lkad, lkcd -> ia', self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT1 += -2.0*np.einsum('kc, ilad, lkcd -> ia', self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT1 += -0.5*np.einsum('la, kicd, klcd -> ia', self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT1 += np.einsum('kc, id, la, lkcd -> ia', self.T1, self.T1, self.T1, self.Voovv, optimize = 'optimal')
        newT1 += 0.5*np.einsum('la, kicd, lkcd -> ia', self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT1 += 0.5*np.einsum('la, ikcd, klcd -> ia', self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT1 += np.einsum('kc, liad, lkcd -> ia', self.T1, self.T2, self.Voovv, optimize = 'optimal')

        # T2 Amplitude equations
        newT2 += self.Voovv
        newT2 += np.einsum('cb, ijac -> ijab', self.fock_VV, self.T2, optimize = 'optimal')
        newT2 -= np.einsum('ik, kjab -> ijab', self.fock_OO, self.T2, optimize = 'optimal')
        newT2 -= np.einsum('jk, ikab -> ijab', self.fock_OO, self.T2, optimize = 'optimal')
        newT2 += np.einsum('ca, ijcb -> ijab', self.fock_VV, self.T2, optimize = 'optimal')
        newT2 -= np.einsum('ka, ijkb -> ijab', self.T1, self.Vooov, optimize = 'optimal')
        newT2 += np.einsum('jc, icab -> ijab', self.T1, self.Vovvv, optimize = 'optimal')
        newT2 -= np.einsum('kc, ka, ijcb -> ijab', self.fock_OV, self.T1, self.T2, optimize = 'optimal')
        newT2 -= np.einsum('kc, kb, ijac -> ijab', self.fock_OV, self.T1, self.T2, optimize = 'optimal')
        newT2 -= np.einsum('kc, jc, ikab -> ijab', self.fock_OV, self.T1, self.T2, optimize = 'optimal')
        newT2 += np.einsum('ic, jcba -> ijab', self.T1, self.Vovvv, optimize = 'optimal')
        newT2 -= np.einsum('kb, jika -> ijab', self.T1, self.Vooov, optimize = 'optimal')
        newT2 -= np.einsum('kc, ic, kjab -> ijab', self.fock_OV, self.T1, self.T2, optimize = 'optimal')
        newT2 += 2.0*np.einsum('ikac, kjcb -> ijab', self.T2, self.Voovv, optimize = 'optimal')
        newT2 -= np.einsum('ikcb, jcka -> ijab', self.T2, self.Vovov, optimize = 'optimal')
        newT2 += np.einsum('ka, lb, ijkl -> ijab', self.T1, self.T1, self.Voooo, optimize = 'optimal')
        newT2 += np.einsum('ic, jd, cdab -> ijab', self.T1, self.T1, self.Vvvvv, optimize = 'optimal')
        newT2 -= np.einsum('jc, kb, kica -> ijab', self.T1, self.T1, self.Voovv, optimize = 'optimal')
        newT2 -= np.einsum('kiac, kjcb -> ijab', self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('lkab, ijlk -> ijab', self.T2, self.Voooo, optimize = 'optimal')
        newT2 -= np.einsum('kjcb, icka -> ijab', self.T2, self.Vovov, optimize = 'optimal')
        newT2 -= np.einsum('jc, ka, ickb -> ijab', self.T1, self.T1, self.Vovov, optimize = 'optimal')
        newT2 -= np.einsum('ic, ka, kjcb -> ijab', self.T1, self.T1, self.Voovv, optimize = 'optimal')
        newT2 -= np.einsum('ic, kb, jcka -> ijab', self.T1, self.T1, self.Vovov, optimize = 'optimal')
        newT2 += np.einsum('ijcd, cdab -> ijab', self.T2, self.Vvvvv, optimize = 'optimal')
        newT2 -= np.einsum('kjac, ickb -> ijab', self.T2, self.Vovov, optimize = 'optimal')
        newT2 += np.einsum('kjcb, kica -> ijab', self.T2, self.Voovv, optimize = 'optimal')
        newT2 -= np.einsum('kjbc, kica -> ijab', self.T2, self.Voovv, optimize = 'optimal')
        newT2 -= np.einsum('ikac, jckb -> ijab', self.T2, self.Vovov, optimize = 'optimal')
        newT2 += np.einsum('jkbc, kica -> ijab', self.T2, self.Voovv, optimize = 'optimal')
        newT2 += 2.0*np.einsum('kc, ijdb, kacd -> ijab', self.T1, self.T2, self.Vovvv, optimize = 'optimal')
        newT2 -= np.einsum('ic, kjdb, kacd -> ijab', self.T1, self.T2, self.Vovvv, optimize = 'optimal')
        newT2 -= np.einsum('ka, ijdc, kbdc -> ijab', self.T1, self.T2, self.Vovvv, optimize = 'optimal')
        newT2 -= np.einsum('ic, kjbd, kadc -> ijab', self.T1, self.T2, self.Vovvv, optimize = 'optimal')
        newT2 += np.einsum('jc, ka, lb, klic -> ijab', self.T1, self.T1, self.T1, self.Vooov, optimize = 'optimal')
        newT2 += np.einsum('lb, ikac, kljc -> ijab', self.T1, self.T2, self.Vooov, optimize = 'optimal')
        newT2 -= np.einsum('ka, ljcb, klic -> ijab', self.T1, self.T2, self.Vooov, optimize = 'optimal')
        newT2 -= np.einsum('jc, kiad, kbdc -> ijab', self.T1, self.T2, self.Vovvv, optimize = 'optimal')
        newT2 -= np.einsum('kc, ijdb, kadc -> ijab', self.T1, self.T2, self.Vovvv, optimize = 'optimal')
        newT2 += np.einsum('ic, ka, lb, lkjc -> ijab', self.T1, self.T1, self.T1, self.Vooov, optimize = 'optimal')
        newT2 += -2.0*np.einsum('kc, ilab, lkjc -> ijab', self.T1, self.T2, self.Vooov, optimize = 'optimal')
        newT2 -= np.einsum('jc, ikdb, kacd -> ijab', self.T1, self.T2, self.Vovvv, optimize = 'optimal')
        newT2 += np.einsum('ic, lkab, kljc -> ijab', self.T1, self.T2, self.Vooov, optimize = 'optimal')
        newT2 -= np.einsum('jc, ikad, kbcd -> ijab', self.T1, self.T2, self.Vovvv, optimize = 'optimal')
        newT2 += np.einsum('kc, ilab, kljc -> ijab', self.T1, self.T2, self.Vooov, optimize = 'optimal')
        newT2 += np.einsum('lb, kiac, lkjc -> ijab', self.T1, self.T2, self.Vooov, optimize = 'optimal')
        newT2 -= np.einsum('ka, jlbc, klic -> ijab', self.T1, self.T2, self.Vooov, optimize = 'optimal')
        newT2 += np.einsum('ka, ilcb, lkjc -> ijab', self.T1, self.T2, self.Vooov, optimize = 'optimal')
        newT2 += np.einsum('lb, kjac, klic -> ijab', self.T1, self.T2, self.Vooov, optimize = 'optimal')
        newT2 -= np.einsum('ic, jd, kb, kadc -> ijab', self.T1, self.T1, self.T1, self.Vovvv, optimize = 'optimal')
        newT2 -= np.einsum('kb, ijcd, kadc -> ijab', self.T1, self.T2, self.Vovvv, optimize = 'optimal')
        newT2 += np.einsum('ic, jkbd, kadc -> ijab', self.T1, self.T2, self.Vovvv, optimize = 'optimal')
        newT2 += -2.0*np.einsum('lb, ikac, lkjc -> ijab', self.T1, self.T2, self.Vooov, optimize = 'optimal')
        newT2 += 2.0*np.einsum('jc, ikad, kbdc -> ijab', self.T1, self.T2, self.Vovvv, optimize = 'optimal')
        newT2 += np.einsum('ka, ljcb, lkic -> ijab', self.T1, self.T2, self.Vooov, optimize = 'optimal')
        newT2 += -2.0*np.einsum('kc, ljab, lkic -> ijab', self.T1, self.T2, self.Vooov, optimize = 'optimal')
        newT2 += 2.0*np.einsum('kc, ijad, kbcd -> ijab', self.T1, self.T2, self.Vovvv, optimize = 'optimal')
        newT2 += np.einsum('ic, kjdb, kadc -> ijab', self.T1, self.T2, self.Vovvv, optimize = 'optimal')
        newT2 -= np.einsum('ic, jd, ka, kbcd -> ijab', self.T1, self.T1, self.T1, self.Vovvv, optimize = 'optimal')
        newT2 += np.einsum('ka, ljbc, klic -> ijab', self.T1, self.T2, self.Vooov, optimize = 'optimal')
        newT2 -= np.einsum('ic, kjad, kbcd -> ijab', self.T1, self.T2, self.Vovvv, optimize = 'optimal')
        newT2 -= np.einsum('kc, ijad, kbdc -> ijab', self.T1, self.T2, self.Vovvv, optimize = 'optimal')
        newT2 += np.einsum('jc, klab, klic -> ijab', self.T1, self.T2, self.Vooov, optimize = 'optimal')
        newT2 += np.einsum('kc, ljab, klic -> ijab', self.T1, self.T2, self.Vooov, optimize = 'optimal')
        newT2 += -2.0*np.einsum('jc, lb, ikad, lkcd -> ijab', self.T1, self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT2 -= np.einsum('ijac, lkbd, lkcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('kc, jd, ilab, lkcd -> ijab', self.T1, self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += -2.0*np.einsum('ikac, ljbd, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('jc, lb, kiad, lkcd -> ijab', self.T1, self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += -2.0*np.einsum('kc, la, ijdb, klcd -> ijab', self.T1, self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT2 -= np.einsum('ic, ka, ljdb, klcd -> ijab', self.T1, self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('ijdc, lkab, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += -0.5*np.einsum('kicd, ljab, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += 0.5*np.einsum('ikcd, ljab, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('ic, ka, ljbd, klcd -> ijab', self.T1, self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('jc, ka, ildb, lkcd -> ijab', self.T1, self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT2 -= np.einsum('ijac, kldb, lkcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('ikac, ljbd, lkcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 -= np.einsum('kiac, ljdb, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += -2.0*np.einsum('lkac, ijdb, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('kc, lb, ijad, lkcd -> ijab', self.T1, self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('kiac, ljdb, lkcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += 0.5*np.einsum('jkcd, ilab, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('kc, id, ljab, lkcd -> ijab', self.T1, self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += -2.0*np.einsum('kc, jd, ilab, klcd -> ijab', self.T1, self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('kiac, ljbd, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += -1.5*np.einsum('ikdc, ljab, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('ic, lb, kjad, klcd -> ijab', self.T1, self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT2 -= np.einsum('ikac, ljdb, lkcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += -2.0*np.einsum('kc, id, ljab, klcd -> ijab', self.T1, self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += 0.5*np.einsum('kicd, ljab, lkcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('ic, jd, klab, klcd -> ijab', self.T1, self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('ka, lb, ijcd, klcd -> ijab', self.T1, self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += -0.5*np.einsum('jkcd, ilab, lkcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('jc, lb, ikad, klcd -> ijab', self.T1, self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += -2.0*np.einsum('kc, lb, ijad, klcd -> ijab', self.T1, self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += -1.5*np.einsum('kjcd, ilab, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += 0.5*np.einsum('kjcd, ilab, lkcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 -= np.einsum('ic, ka, jlbd, klcd -> ijab', self.T1, self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('kc, la, ijdb, lkcd -> ijab', self.T1, self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('ijac, klbd, lkcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('ic, ka, ljdb, lkcd -> ijab', self.T1, self.T1, self.T2, self.Voovv, optimize = 'optimal')
        newT2 -= np.einsum('kiac, jlbd, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 -= np.einsum('ikac, jlbd, lkcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += 2.0*np.einsum('ikac, ljdb, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('klac, ijdb, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('kjac, ildb, lkcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += 2.0*np.einsum('ikac, jlbd, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('ic, jd, ka, lb, klcd -> ijab', self.T1, self.T1, self.T1, self.T1, self.Voovv, optimize = 'optimal')

        newT1 *= self.d
        newT2 *= self.D

        # Compute RMS

        self.rms1 = np.sqrt(np.sum(np.square(newT1 - self.T1 )))/(self.ndocc*self.nvir)
        self.rms2 = np.sqrt(np.sum(np.square(newT2 - self.T2 )))/(self.ndocc*self.ndocc*self.nvir*self.nvir)

        # Save new amplitudes

        self.T1 = newT1
        self.T2 = newT2

    def __init__(self, wfn, CC_CONV=6, CC_MAXITER=50, E_CONV=8):

        # Save reference wavefunction properties
        self.Ehf = wfn.energy() 
        self.nmo = wfn.nmo()
        self.nelec = wfn.nalpha() + wfn.nbeta()
        if self.nelec % 2 != 0:
            NameError('Invalid number of electrons for RHF') 
        self.ndocc = int(self.nelec/2)
        self.nvir = self.nmo - self.ndocc
        self.C = wfn.Ca()
        self.Vnuc = wfn.molecule().nuclear_repulsion_energy()

        # Save Options
        self.CC_CONV = CC_CONV
        self.E_CONV = E_CONV
        self.CC_MAXITER = CC_MAXITER

        print("Number of electrons:              {}".format(self.nelec))
        print("Number of Doubly Occupied MOs:    {}".format(self.ndocc))
        print("Number of MOs:                    {}".format(self.nmo))

        print("\n Transforming integrals...")

        mints = psi4.core.MintsHelper(wfn.basisset())
        self.mints = mints
        # One electron integral
        h = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
        ## Alpha
        h = np.einsum('up,vq,uv->pq', self.C, self.C, h)
        Vchem = np.asarray(mints.mo_eri(self.C, self.C, self.C, self.C))
    
        # Slices
        o = slice(0, self.ndocc)
        v = slice(self.ndocc, self.nmo)

        # Form the full fock matrices
        f = h + 2*np.einsum('pqkk->pq', Vchem[:,:,o,o]) - np.einsum('pkqk->pq', Vchem[:,o,:,o])

        # Save diagonal terms
        self.fock_Od = copy.deepcopy(f.diagonal()[o])
        self.fock_Vd = copy.deepcopy(f.diagonal()[v])

        # Erase diagonal elements from original matrix
        np.fill_diagonal(f, 0.0)

        # Save useful slices
        self.fock_OO = f[o,o]
        self.fock_VV = f[v,v]
        self.fock_OV = f[o,v]

        # Save slices of two-electron repulsion integral
        Vphys = Vchem.swapaxes(1,2)
        Vsa = 2*Vphys - Vphys.swapaxes(2,3)

        self.Aovvv = Vsa[o,v,v,v]
        self.Aooov = Vsa[o,o,o,v]
        self.Aoovv = Vsa[o,o,v,v]
        self.Avoov = Vsa[v,o,o,v]

        self.Voooo = Vphys[o,o,o,o]
        self.Vooov = Vphys[o,o,o,v]
        self.Voovv = Vphys[o,o,v,v]
        self.Vovov = Vphys[o,v,o,v]
        self.Vovvv = Vphys[o,v,v,v]
        self.Vvvvv = Vphys[v,v,v,v]

        self.compute()

    def compute(self):

        # Auxiliar D matrix

        new = np.newaxis
        self.d = 1.0/(self.fock_Od[:, new] - self.fock_Vd[new, :])
        self.D = 1.0/(self.fock_Od[:, new, new, new] + self.fock_Od[new, :, new, new] - self.fock_Vd[new, new, :, new] - self.fock_Vd[new, new, new, :])

        # Initial T1 amplitudes

        self.T1 = self.fock_OV*self.d

        # Initial T2 amplitudes

        self.T2 = self.D*self.Voovv

        # Get MP2 energy

        self.update_energy()

        print('MP2 Energy:   {:<15.10f}'.format(self.Ecc + self.Ehf))

        # Setup iteration options
        max_rms = 1
        dE = 1
        ite = 1
        rms_LIM = 10**(-self.CC_CONV)
        E_LIM = 10**(-self.E_CONV)
        t0 = time.time()
        print('='*37)

        # Start CC iterations
        while abs(dE) > E_LIM or max_rms > rms_LIM:
            t = time.time()
            if ite > self.CC_MAXITER:
                raise NameError('CC equations did not converge')
            self.update_amp()
            dE = -self.Ecc
            self.update_energy()
            dE += self.Ecc
            max_rms = max(self.rms1, self.rms2)
            print("Iteration {}".format(ite))
            print("CC Correlation energy: {:< 15.10f}".format(self.Ecc))
            print("Energy change:         {:< 15.10f}".format(dE))
            print("Max RMS residue:       {:< 15.10f}".format(max_rms))
            print("Time required:         {:< 15.10f}".format(time.time() - t))
            print('='*37)
            ite += 1

        print('CC Energy:   {:<15.10f}'.format(self.Ecc + self.Ehf))
        print('CCD iterations took %.2f seconds.\n' % (time.time() - t0))
