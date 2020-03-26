import psi4
import os
import sys
import numpy as np
import time
import copy
np.set_printoptions(suppress=True, linewidth=120)

def update_energy(T2, Voovv):

    # Note that these equations are obtianed automatically using Sympy and Tchau-Spin.
    # See the getting_CCD.ipynb notebook

    CC_energy = 0
    B_OoVv = -1.0*copy.deepcopy(T2)
    B_OoVv += 2.0*T2.transpose(1,0,2,3)
    CC_energy += np.einsum('lkcd, klcd -> ', B_OoVv, Voovv, optimize = 'optimal')
    return CC_energy

def update_amp(T2, f, V, D, info):

    Voooo, Vooov, Voovv, Vovov, Vovvv, Vvvvv = V
    fock_OO, fock_OV, fock_VV = f

    newT2 = np.zeros((info['ndocc'],info['ndocc'],info['nvir'],info['nvir']))

    # Equations from Tchau-spin

    newT2 += Voovv
    newT2 += np.einsum('ijcd, cdab -> ijab', T2, Vvvvv, optimize = 'optimal')
    newT2 += np.einsum('klab, ijkl -> ijab', T2, Voooo, optimize = 'optimal')
    newT2 += np.einsum('kjac, ildb, lkcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += -2.0*np.einsum('ijac, klbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('ijac, lkbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += -2.0*np.einsum('kiac, jlbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('kiac, ljbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('ikac, ljbd, lkcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += 4.0*np.einsum('ikac, jlbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += -2.0*np.einsum('ikac, ljbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('klac, ijdb, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += -2.0*np.einsum('lkac, ijdb, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('ijdc, lkab, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('kiac, ljdb, lkcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += -2.0*np.einsum('ikac, jlbd, lkcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    P_OOVV = 1.0*np.einsum('cb, ijac -> ijab', fock_VV, T2, optimize = 'optimal')
    P_OOVV += -1.0*np.einsum('ik, kjab -> ijab', fock_OO, T2, optimize = 'optimal')
    P_OOVV += 2.0*np.einsum('jkbc, kica -> ijab', T2, Voovv, optimize = 'optimal')
    P_OOVV += -1.0*np.einsum('kjbc, kica -> ijab', T2, Voovv, optimize = 'optimal')
    P_OOVV += -1.0*np.einsum('ikcb, jcka -> ijab', T2, Vovov, optimize = 'optimal')
    P_OOVV += -1.0*np.einsum('ikac, jckb -> ijab', T2, Vovov, optimize = 'optimal')
    P_OOVV += 1.0*np.einsum('jkcd, ilab, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    P_OOVV += -2.0*np.einsum('kjcd, ilab, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    
    newT2 += P_OOVV + P_OOVV.transpose(1,0,3,2)

    newT2 *= D

    # Compute RMS

    rms = np.sqrt(np.sum(np.square(newT2 - T2 )))/(info['ndocc']*info['ndocc']*info['nvir']*info['nvir'])

    return newT2, rms

def get_integrals(wfn, info):

    print("\n Transforming integrals...")

    C = wfn.Ca()
    mints = psi4.core.MintsHelper(wfn.basisset())
    mints = mints
    # One electron integral
    h = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
    ## Alpha
    h = np.einsum('up,vq,uv->pq', C, C, h)
    Vchem = np.asarray(mints.mo_eri(C, C, C, C))

    # Slices
    o = slice(0, info['ndocc'])
    v = slice(info['ndocc'], info['nmo'])

    # Form the full fock matrices
    f = h + 2*np.einsum('pqkk->pq', Vchem[:,:,o,o]) - np.einsum('pkqk->pq', Vchem[:,o,:,o])

    # Save diagonal terms
    fock_Od = copy.deepcopy(f.diagonal()[o])
    fock_Vd = copy.deepcopy(f.diagonal()[v])
    fd = (fock_Od, fock_Vd)

    # Erase diagonal elements from original matrix
    np.fill_diagonal(f, 0.0)

    # Save useful slices
    fock_OO = f[o,o]
    fock_VV = f[v,v]
    fock_OV = f[o,v]
    f = (fock_OO, fock_OV, fock_VV)

    # Save slices of two-electron repulsion integral
    Vphys = Vchem.swapaxes(1,2)

    V = (Vphys[o,o,o,o], Vphys[o,o,o,v], Vphys[o,o,v,v], Vphys[o,v,o,v], Vphys[o,v,v,v], Vphys[v,v,v,v])

    return fd, f, V

def RCCD(wfn, CC_CONV=6, CC_MAXITER=50, E_CONV=8):

    # Save reference wavefunction properties
    info = {}
    Ehf = wfn.energy() 
    info['nmo'] = wfn.nmo()
    info['nelec'] = wfn.nalpha() + wfn.nbeta()
    if info['nelec'] % 2 != 0:
        NameError('Invalid number of electrons for RHF') 
    info['ndocc'] = int(info['nelec']/2)
    info['nvir'] = info['nmo'] - info['ndocc']

    # Save Options
    CC_CONV = CC_CONV
    E_CONV = E_CONV
    CC_MAXITER = CC_MAXITER

    print("Number of electrons:              {}".format(info['nelec']))
    print("Number of Doubly Occupied MOs:    {}".format(info['ndocc']))
    print("Number of MOs:                    {}".format(info['nmo']))

    fd, f, V = get_integrals(wfn, info)

    # Auxiliar D matrix

    fock_Od, fock_Vd = fd
    new = np.newaxis
    D = 1.0/(fock_Od[:, new, new, new] + fock_Od[new, :, new, new] - fock_Vd[new, new, :, new] - fock_Vd[new, new, new, :])

    # Initial T2 amplitudes

    T2 = D*V[2]

    # Get MP2 energy

    Ecc = update_energy(T2, V[2])

    print('MP2 Energy:   {:<15.10f}'.format(Ecc + Ehf))

    # Setup iteration options
    rms = 0.0
    dE = 1
    ite = 1
    rms_LIM = 10**(-CC_CONV)
    E_LIM = 10**(-E_CONV)
    t0 = time.time()
    print('='*37)

    # Start CC iterations
    while abs(dE) > E_LIM or rms > rms_LIM:
        t = time.time()
        if ite > CC_MAXITER:
            raise NameError('CC equations did not converge')
        T2, rms = update_amp(T2, f, V, D, info)
        dE = -Ecc
        Ecc = update_energy(T2, V[2])
        dE += Ecc
        print("Iteration {}".format(ite))
        print("CC Correlation energy: {:< 15.10f}".format(Ecc))
        print("Energy change:         {:< 15.10f}".format(dE))
        print("Max RMS residue:       {:< 15.10f}".format(rms))
        print("Time required:         {:< 15.10f}".format(time.time() - t))
        print('='*37)
        ite += 1

    print('CC Energy:   {:<15.10f}'.format(Ecc + Ehf))
    print('CCD iterations took %.2f seconds.\n' % (time.time() - t0))
    return (Ecc + Ehf)

if __name__ == '__main__':
    
    psi4.core.be_quiet()
    mol = psi4.geometry("""
    0 1
    O
    H 1 0.96
    H 1 0.96 2 104.5
    symmetry c1""")
    mol.update_geometry()

    psi4.set_options({
        'basis' : 'cc-pvtz',
        'scf_type' : 'pk',
        'e_convergence': 1e-12,
        'reference' : 'rhf'})

    Ehf, wfn = psi4.energy('scf', return_wfn = True)
    RCCD(wfn, CC_CONV = 8, E_CONV = 12)
