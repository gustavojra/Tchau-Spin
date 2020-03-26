from sympy.physics.secondquant import (AntiSymmetricTensor, wicks,
        F, Fd, NO, evaluate_deltas, substitute_dummies, Commutator,
        simplify_index_permutations, PermutationOperator)
from sympy import (
    symbols, Rational, latex, Dummy
)

# For Sympy (simplification of equations)
pretty_dummies_dict = {
    'above': 'defgh',
    'below': 'lmnow',
    'general': 'pqrstu'
}


i = symbols('i', below_fermi=True, cls=Dummy)
a = symbols('a', above_fermi=True, cls=Dummy)
j = symbols('j', below_fermi=True, cls=Dummy)
b = symbols('b', above_fermi=True, cls=Dummy)
p, q, r, s = symbols('p,q,r,s', cls=Dummy)


## Build the $\Phi$-normal ordered Hamiltonian

fock = AntiSymmetricTensor('f', (p,), (q,))
pr = NO(Fd(p)*F(q))
V = AntiSymmetricTensor('v',(p,q),(r,s))
pqsr = NO(Fd(p)*Fd(q)*F(s)*F(r))
H = fock*pr + Rational(1,4)*V*pqsr

# ## Build Cluster Operator

def get_T():
    i, j, k = symbols('i,j,k', below_fermi=True, cls=Dummy)
    a, b, c = symbols('a,b,c', above_fermi=True, cls=Dummy)
    t1 = AntiSymmetricTensor('t', (a,), (i,))*NO(Fd(a)*F(i))
    t2 = Rational(1,4)*AntiSymmetricTensor('t', (a,b), (i,j))*NO(Fd(a)*Fd(b)*F(j)*F(i))
    t3 = Rational(1,36)*AntiSymmetricTensor('t', (a,b,c), (i,j,k))*NO(Fd(a)*Fd(b)*Fd(c)*F(k)*F(j)*F(i))
    return t1 + t2 + t3



# ## Perform the Hausdorff expansion

C = Commutator
T = get_T()
print("commutator 1...")
comm1 = wicks(C(H, T))
comm1 = evaluate_deltas(comm1)
comm1 = substitute_dummies(comm1)

T = get_T()
print("commutator 2...")
comm2 = wicks(C(comm1, T))
comm2 = evaluate_deltas(comm2)
comm2 = substitute_dummies(comm2)

T = get_T()
print("commutator 3...")
comm3 = wicks(C(comm2, T))
comm3 = evaluate_deltas(comm3)
comm3 = substitute_dummies(comm3)

T = get_T()
print("commutator 4...")
comm4 = wicks(C(comm3, T))
comm4 = evaluate_deltas(comm4)
comm4 = substitute_dummies(comm4)

# ## Construct the Similarity Transformed Hamiltonian

eq = H + comm1 + comm2/2 + comm3/6 + comm4/24
eq = eq.expand()
eq = evaluate_deltas(eq)
eq = substitute_dummies(eq, new_indices=True,
        pretty_indices=pretty_dummies_dict)


# ## Get energy expression

i, j, k, l = symbols('i,j,k,l', below_fermi=True)
a, b, c, d = symbols('a,b,c,d', above_fermi=True)
energy = wicks(eq, simplify_dummies=True,
        keep_only_fully_contracted=True)

# # Get $T_1$ amplitude equation

eqT1 = wicks(NO(Fd(i)*F(a))*eq, simplify_dummies=True, 
             keep_only_fully_contracted=True, simplify_kronecker_deltas=True)

# ## Get $T_2$ amplitude equation


eqT2 = wicks(NO(Fd(i)*Fd(j)*F(b)*F(a))*eq, simplify_dummies=True, 
             keep_only_fully_contracted=True, simplify_kronecker_deltas=True)

# ## Get $T_3$ amplitude equation

eqT3 = wicks(NO(Fd(i)*Fd(j)*Fd(k)*F(c)*F(b)*F(a))*eq, simplify_dummies=True, 
             keep_only_fully_contracted=True, simplify_kronecker_deltas=True)

# # Using Tchau-Spin

import sys
sys.path.append('..')
from tchau_spin import *
# For restricted CC
Tensor.rhf = True


# $\bullet$ Define indexes

i,j,k,a,b,c = Index.new('ijkabc', 'abaaba', 'hhhppp')
# 'xxxx' indicates that those indexes have any spin
l,m,d,e = Index.new('lmde', 'xxxx', 'hhpp')
index_key = {'i':i,
             'j':j,
             'a':a,
             'b':b,
             'k':k,
             'l':l,
             'c':c,
             'd':d,
             'l':l,
             'm':m,
             'e':e
}

mytensors = {
    'T_OoVv' : 'T2',
    'T_OV' : 'T1',
    'T_OoOVvV': 'T3',
    'V_OOOO' : 'Voooo',
    'V_OOOV' : 'Vooov',
    'V_OOVV' : 'Voovv',
    'V_OVOV' : 'Vovov',
    'V_OVVV' : 'Vovvv',
    'V_VVVV' : 'Vvvvv',
    'f_OO' : 'fock_OO',
    'f_VV' : 'fock_VV',
    'f_OV' : 'fock_OV'
    
}

# ## Processing Energy expression

E = eqfromlatex(latex(energy), index_key)

# $\bullet$ Simplify Equation

print('Simplifying Energy')
E = E.simplify()
E = E.adapt_space()
E = Factor.factorize_ERI(E)

pE = process_eq(eq=E, name = 'CC_energy', tensor_labels = mytensors)
pE.write_einsums_out(output='full_T_energy.dat')
print('Energy Ok')


# ## Processing $T_1$ amplitude equation

T1 = eqfromlatex(latex(eqT1), index_key)

print('Simplifying T1')
T1 = T1.simplify(show_progress=True)
T1 = T1.adapt_space()
T1.sort()

pT1 = process_eq(T1,i,a,name = 'newT1', tensor_labels = mytensors)
pT1.write_einsums_out(output='full_T_T1.dat')

# ## Processing $T_2$ amplitude equation

T2 = eqfromlatex(latex(eqT2), index_key)

print('Simplifying T2')
T2 = T2.simplify(show_progress=True)
T2 = T2.adapt_space()
T2 = Permutation.find_permutations(T2, (i,j), (a,b), show_progress=True)
T2.sort()

pT2 = process_eq(T2,i,j,a,b, name = 'newT2', tensor_labels = mytensors)
pT2.write_einsums_out(output='full_T_T2.dat')

# ## Processing $T_3$ amplitude equation

T3 = eqfromlatex(latex(eqT3), index_key)

print('Simplifying T3')
T3 = T3.simplify(show_progress=True)
T3 = T3.adapt_space()
T3.sort()

pT3 = process_eq(T3,i,j,k,a,b,c, name = 'newT3', tensor_labels = mytensors)
pT3.write_einsums_out(output='full_T_T3_no_perm.dat')

perms = [\
 [(i,j)],
 [(i,k)],
 [(j,k)],
 [(a,b)],
 [(a,c)],
 [(b,c)]] 

for p in perms:
    T3 = Permutation.find_permutations(T3, *p, show_progress=True)

for i,p1 in enumerate(perms):
    for j,p2 in enumerate(perms):
        if i >= j:
            continue
        else:
            T3 = Permutation.find_permutations(T3, p1, p2, show_progress=True)
            
pT3 = process_eq(T3,i,j,k,a,b,c, name = 'newT3', tensor_labels = mytensors)
pT3.write_einsums_out(output='full_T_T3.dat')
