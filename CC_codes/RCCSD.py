import psi4
import os
import sys
import numpy as np
import time
import copy
np.set_printoptions(suppress=True, linewidth=120)

def update_energy(T1, T2, fock_OV, Voovv):

    # Note that these equations are obtianed automatically using Sympy and Tchau-Spin.
    # See the getting_CCSD.ipynb notebook

    CC_energy = 0
    CC_energy += 2.0*np.einsum('kc, kc -> ', fock_OV, T1, optimize = 'optimal')
    B_OVOV = -1.0*np.einsum('lc, kd -> lckd', T1, T1, optimize = 'optimal')
    B_OVOV += -1.0*T2.transpose(0,2,1,3)
    B_OVOV += 2.0*T2.transpose(1,2,0,3)
    CC_energy += np.einsum('lckd, klcd -> ', B_OVOV, Voovv, optimize = 'optimal')
    CC_energy += 2.0*np.einsum('lc, kd, lkcd -> ', T1, T1, Voovv, optimize = 'optimal')
    return CC_energy

def update_amp(T1, T2, f, V, d, D, info):

    Voooo, Vooov, Voovv, Vovov, Vovvv, Vvvvv = V
    fock_OO, fock_OV, fock_VV = f

    newT1 = np.zeros(T1.shape)
    newT2 = np.zeros(T2.shape)

    # T1 equation
    newT1 += fock_OV
    newT1 -= np.einsum('ik, ka -> ia', fock_OO, T1, optimize = 'optimal')
    newT1 += np.einsum('ca, ic -> ia', fock_VV, T1, optimize = 'optimal')
    newT1 -= np.einsum('kc, ic, ka -> ia', fock_OV, T1, T1, optimize = 'optimal')
    newT1 += 2.0*np.einsum('kc, ikac -> ia', fock_OV, T2, optimize = 'optimal')
    newT1 -= np.einsum('kc, kiac -> ia', fock_OV, T2, optimize = 'optimal')
    newT1 -= np.einsum('kc, icka -> ia', T1, Vovov, optimize = 'optimal')
    newT1 += 2.0*np.einsum('kc, kica -> ia', T1, Voovv, optimize = 'optimal')
    newT1 -= np.einsum('kicd, kadc -> ia', T2, Vovvv, optimize = 'optimal')
    newT1 += 2.0*np.einsum('ikcd, kadc -> ia', T2, Vovvv, optimize = 'optimal')
    newT1 += -2.0*np.einsum('klac, klic -> ia', T2, Vooov, optimize = 'optimal')
    newT1 += np.einsum('lkac, klic -> ia', T2, Vooov, optimize = 'optimal')
    newT1 += -2.0*np.einsum('kc, la, lkic -> ia', T1, T1, Vooov, optimize = 'optimal')
    newT1 -= np.einsum('kc, id, kadc -> ia', T1, T1, Vovvv, optimize = 'optimal')
    newT1 += 2.0*np.einsum('kc, id, kacd -> ia', T1, T1, Vovvv, optimize = 'optimal')
    newT1 += np.einsum('kc, la, klic -> ia', T1, T1, Vooov, optimize = 'optimal')
    newT1 += -2.0*np.einsum('kc, ilad, lkcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += -2.0*np.einsum('kc, liad, klcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += np.einsum('kc, liad, lkcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += -2.0*np.einsum('ic, lkad, lkcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += np.einsum('ic, lkad, klcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += -2.0*np.einsum('la, ikdc, klcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += np.einsum('la, ikcd, klcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += np.einsum('kc, id, la, lkcd -> ia', T1, T1, T1, Voovv, optimize = 'optimal')
    newT1 += -2.0*np.einsum('kc, id, la, klcd -> ia', T1, T1, T1, Voovv, optimize = 'optimal')
    newT1 += 4.0*np.einsum('kc, ilad, klcd -> ia', T1, T2, Voovv, optimize = 'optimal')

    # T2 equation
    newT2 += Voovv
    newT2 += np.einsum('ic, jd, cdab -> ijab', T1, T1, Vvvvv, optimize = 'optimal')
    newT2 += np.einsum('ijcd, cdab -> ijab', T2, Vvvvv, optimize = 'optimal')
    newT2 += np.einsum('ka, lb, ijkl -> ijab', T1, T1, Voooo, optimize = 'optimal')
    newT2 += np.einsum('klab, ijkl -> ijab', T2, Voooo, optimize = 'optimal')
    newT2 -= np.einsum('ic, jd, ka, kbcd -> ijab', T1, T1, T1, Vovvv, optimize = 'optimal')
    newT2 -= np.einsum('ic, jd, kb, kadc -> ijab', T1, T1, T1, Vovvv, optimize = 'optimal')
    newT2 += np.einsum('ic, ka, lb, lkjc -> ijab', T1, T1, T1, Vooov, optimize = 'optimal')
    newT2 += np.einsum('jc, ka, lb, klic -> ijab', T1, T1, T1, Vooov, optimize = 'optimal')
    newT2 += np.einsum('klac, ijdb, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += -2.0*np.einsum('ikac, ljbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += -2.0*np.einsum('lkac, ijdb, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('kiac, ljdb, lkcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('ikac, ljbd, lkcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += -2.0*np.einsum('ikac, jlbd, lkcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('kiac, ljbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += -2.0*np.einsum('kiac, jlbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('ijac, lkbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += -2.0*np.einsum('ijac, klbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('kjac, ildb, lkcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += 4.0*np.einsum('ikac, jlbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('ijdc, lkab, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('ic, jd, ka, lb, klcd -> ijab', T1, T1, T1, T1, Voovv, optimize = 'optimal')
    newT2 += np.einsum('ic, jd, lkab, lkcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('ka, lb, ijdc, lkcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO = -1.0*np.einsum('ik, kjab -> ijab', fock_OO, T2, optimize = 'optimal')
    P_OVVO += 1.0*np.einsum('ca, ijcb -> ijab', fock_VV, T2, optimize = 'optimal')
    P_OVVO += -1.0*np.einsum('kb, jika -> ijab', T1, Vooov, optimize = 'optimal')
    P_OVVO += 1.0*np.einsum('jc, icab -> ijab', T1, Vovvv, optimize = 'optimal')
    P_OVVO += -1.0*np.einsum('kc, ic, kjab -> ijab', fock_OV, T1, T2, optimize = 'optimal')
    P_OVVO += -1.0*np.einsum('kc, ka, ijcb -> ijab', fock_OV, T1, T2, optimize = 'optimal')
    P_OVVO += -1.0*np.einsum('kiac, kjcb -> ijab', T2, Voovv, optimize = 'optimal')
    P_OVVO += -1.0*np.einsum('ic, ka, kjcb -> ijab', T1, T1, Voovv, optimize = 'optimal')
    P_OVVO += -1.0*np.einsum('ic, kb, jcka -> ijab', T1, T1, Vovov, optimize = 'optimal')
    P_OVVO += 2.0*np.einsum('ikac, kjcb -> ijab', T2, Voovv, optimize = 'optimal')
    P_OVVO += -1.0*np.einsum('ikac, jckb -> ijab', T2, Vovov, optimize = 'optimal')
    P_OVVO += -1.0*np.einsum('kjac, ickb -> ijab', T2, Vovov, optimize = 'optimal')
    P_OVVO += -2.0*np.einsum('lb, ikac, lkjc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO += 1.0*np.einsum('lb, kiac, lkjc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO += -1.0*np.einsum('jc, ikdb, kacd -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO += -1.0*np.einsum('jc, kiad, kbdc -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO += -1.0*np.einsum('jc, ikad, kbcd -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO += 1.0*np.einsum('jc, lkab, lkic -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO += 1.0*np.einsum('lb, ikac, kljc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO += -1.0*np.einsum('ka, ijdc, kbdc -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO += 1.0*np.einsum('ka, ilcb, lkjc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO += 2.0*np.einsum('jc, ikad, kbdc -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO += -1.0*np.einsum('kc, ijad, kbdc -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO += 2.0*np.einsum('kc, ijad, kbcd -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO += 1.0*np.einsum('kc, ilab, kljc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO += -2.0*np.einsum('kc, ilab, lkjc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO += 1.0*np.einsum('jkcd, ilab, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    P_OVVO += -2.0*np.einsum('kc, jd, ilab, klcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += 1.0*np.einsum('kc, jd, ilab, lkcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += -2.0*np.einsum('kc, la, ijdb, klcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += 1.0*np.einsum('kc, la, ijdb, lkcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += 1.0*np.einsum('ic, ka, ljbd, klcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += -2.0*np.einsum('ic, ka, jlbd, klcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += 1.0*np.einsum('ic, ka, ljdb, lkcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += 1.0*np.einsum('ic, lb, kjad, klcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += -2.0*np.einsum('ikdc, ljab, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    
    newT2 += P_OVVO + P_OVVO.transpose(1,0,3,2)

    newT1 *= d
    newT2 *= D

    # Compute RMS

    r1 = np.sqrt(np.sum(np.square(newT1 - T1 )))/(info['ndocc']*info['nvir'])
    r2 = np.sqrt(np.sum(np.square(newT2 - T2 )))/(info['ndocc']*info['ndocc']*info['nvir']*info['nvir'])

    return newT1, newT2, r1, r2

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

def RCCSD(wfn, CC_CONV=6, CC_MAXITER=50, E_CONV=8):

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
    d = 1.0/(fock_Od[:, new] - fock_Vd[new, :])

    # Initial Amplitudes

    T1 = f[1]*d
    T2 = D*V[2]

    # Get MP2 energy

    Ecc = update_energy(T1, T2, f[1], V[2])

    print('MP2 Energy:   {:<15.10f}'.format(Ecc + Ehf))

    # Setup iteration options
    r1 = 1
    r2 = 1
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
        T1, T2, r1, r2 = update_amp(T1, T2, f, V, d, D, info)
        rms = max(r1, r2)
        dE = -Ecc
        Ecc = update_energy(T1, T2, f[1], V[2])
        dE += Ecc
        print("Iteration {}".format(ite))
        print("CC Correlation energy: {:< 15.10f}".format(Ecc))
        print("Energy change:         {:< 15.10f}".format(dE))
        print("Max RMS residue:       {:< 15.10f}".format(rms))
        print("Time required:         {:< 15.10f}".format(time.time() - t))
        print('='*37)
        ite += 1

    print('CCSD Energy:   {:<15.10f}'.format(Ecc + Ehf))
    print('CCSD iterations took %.2f seconds.\n' % (time.time() - t0))
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
        'basis' : 'sto-3g',
        'scf_type' : 'pk',
        'e_convergence': 1e-12,
        'reference' : 'rhf'})

    Ehf, wfn = psi4.energy('scf', return_wfn = True)
    RCCSD(wfn, CC_CONV = 8, E_CONV = 12)
    p4 = psi4.energy('ccsd')
    print('Psi4 Energy:   {:<15.10f}'.format(p4))
