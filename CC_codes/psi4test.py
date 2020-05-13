import psi4

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
    'diis' : False,
    'reference' : 'rhf'})

Ehf, wfn = psi4.energy('ccsd', return_wfn = True)
