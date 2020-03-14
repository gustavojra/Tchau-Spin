import psi4
import os
import sys
import numpy as np
import time
import copy

np.set_printoptions(suppress=True, linewidth=120)

class RCCD:

    def update_energy(self):
        
        self.Ecc = -np.einsum('jiab, ijab -> ', self.T2, self.Voovv, optimize = 'optimal')
        self.Ecc += 2.0*np.einsum('ijab, ijab -> ', self.T2, self.Voovv, optimize = 'optimal') 

    def update_amp(self):

        newT2 = np.zeros((self.ndocc,self.ndocc,self.nvir,self.nvir))
        newT2 += self.Voovv
        newT2 += np.einsum('ijcd, abcd -> ijab', self.T2, self.Vvvvv, optimize = 'optimal')
        newT2 += np.einsum('klab, ijkl -> ijab', self.T2, self.Voooo, optimize = 'optimal')
        newT2 -= np.einsum('ikac, kbjc -> ijab', self.T2, self.Vovov, optimize = 'optimal')
        newT2 += 2.0*np.einsum('ikac, jkbc -> ijab', self.T2, self.Voovv, optimize = 'optimal')
        newT2 -= np.einsum('kiac, jkbc -> ijab', self.T2, self.Voovv, optimize = 'optimal')
        newT2 -= np.einsum('kjac, kbic -> ijab', self.T2, self.Vovov, optimize = 'optimal')
        newT2 -= np.einsum('ikcb, kajc -> ijab', self.T2, self.Vovov, optimize = 'optimal')
        newT2 -= np.einsum('kjcb, kaic -> ijab', self.T2, self.Vovov, optimize = 'optimal')
        newT2 -= np.einsum('kjbc, ikac -> ijab', self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('jkbc, ikac -> ijab', self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('kjcb, ikac -> ijab', self.T2, self.Voovv, optimize = 'optimal')
        newT2 -= np.einsum('jk, ikab -> ijab', self.fock_OO, self.T2, optimize = 'optimal')
        newT2 -= np.einsum('ik, kjab -> ijab', self.fock_OO, self.T2, optimize = 'optimal')
        newT2 += np.einsum('bc, ijac -> ijab', self.fock_VV, self.T2, optimize = 'optimal')
        newT2 += np.einsum('ac, ijcb -> ijab', self.fock_VV, self.T2, optimize = 'optimal')
        newT2 += np.einsum('lkab, ijdc, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('lkbd, ijac, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 -= np.einsum('klbd, ijac, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 -= np.einsum('lkdb, ijac, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += -2.0*np.einsum('lkac, ijdb, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('klac, ijdb, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += 0.5*np.einsum('ljcd, ikab, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += -0.5*np.einsum('jlcd, ikab, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += -1.5*np.einsum('ljdc, ikab, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += 0.5*np.einsum('jlcd, ikab, kldc -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += -1.5*np.einsum('ljab, ikdc, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += -0.5*np.einsum('ljab, kicd, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += 0.5*np.einsum('ljab, ikcd, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += 0.5*np.einsum('ljab, kicd, kldc -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += -2.0*np.einsum('ljbd, ikac, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += 2.0*np.einsum('jlbd, ikac, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += 2.0*np.einsum('ljdb, ikac, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('ljbd, kiac, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 -= np.einsum('jlbd, kiac, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 -= np.einsum('ljdb, kiac, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('ljbd, ikac, kldc -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 -= np.einsum('jlbd, ikac, kldc -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('ljdb, kiac, kldc -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 -= np.einsum('ljdb, ikac, kldc -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')
        newT2 += np.einsum('ljac, ikdb, klcd -> ijab', self.T2, self.T2, self.Voovv, optimize = 'optimal')

        newT2 *= self.D

        # Compute RMS

        self.rms = np.sqrt(np.sum(np.square(newT2 - self.T2 )))/(self.ndocc*self.ndocc*self.nvir*self.nvir)

        # Save new amplitudes

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
        self.D = 1.0/(self.fock_Od[:, new, new, new] + self.fock_Od[new, :, new, new] - self.fock_Vd[new, new, :, new] - self.fock_Vd[new, new, new, :])

        # Initial T2 amplitudes

        self.T2 = self.D*self.Voovv

        # Get MP2 energy

        self.update_energy()

        print('MP2 Energy:   {:<15.10f}'.format(self.Ecc + self.Ehf))

        # Setup iteration options
        self.rms = 0.0
        dE = 1
        ite = 1
        rms_LIM = 10**(-self.CC_CONV)
        E_LIM = 10**(-self.E_CONV)
        t0 = time.time()
        print('='*37)

        # Start CC iterations
        while abs(dE) > E_LIM or self.rms > rms_LIM:
            t = time.time()
            if ite > self.CC_MAXITER:
                raise NameError('CC equations did not converge')
            self.update_amp()
            dE = -self.Ecc
            self.update_energy()
            dE += self.Ecc
            print("Iteration {}".format(ite))
            print("CC Correlation energy: {:< 15.10f}".format(self.Ecc))
            print("Energy change:         {:< 15.10f}".format(dE))
            print("Max RMS residue:       {:< 15.10f}".format(self.rms))
            print("Time required:         {:< 15.10f}".format(time.time() - t))
            print('='*37)
            ite += 1

        print('CC Energy:   {:<15.10f}'.format(self.Ecc + self.Ehf))
        print('CCD iterations took %.2f seconds.\n' % (time.time() - t0))
        print('Yesssss')
