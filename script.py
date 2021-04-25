# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 23:26:04 2021

@author: Prathamesh
"""

from solve.solvers import Jacobi, Milaszewicz
from solve.gaussSeidel import GaussSeidel;

from generate.generators import SystemGenerator

if __name__ == '__main__':

    sg = SystemGenerator() # system generator
    
    s = sg.generate(dim=500,kind='positive',diagonally_dominant=True) # generate a system of linear equations
    
    # A,b = s.A,s.b
    
    '-----------------------------------------------Standard Jacobi-----------------------------------------------'
    
    jac1 = Jacobi(s,compute_spectral_radius=True,use_modified_method=False,warm_start=True)
    
    jac1.solve(tol=1e-10,max_iters=2000)
    
    print(jac1,end='\n\n')

    '-----------------------------------------------Standard GaussSeidel-----------------------------------------------'

    gase1 = GaussSeidel(s,compute_spectral_radius=True,use_modified_method=False,warm_start=True)

    gase1.solve(tol=1e-10,max_iters=2000)

    print(gase1,end='\n\n')
    
    '-----------------------------------------------Modified Jacobi-----------------------------------------------'
    
    
    jac2 = Jacobi(s,compute_spectral_radius=True,use_modified_method=True,warm_start=True)
    
    jac2.solve(tol=1e-6,max_iters=2000)
    
    print(jac2,end='\n\n')

    '-----------------------------------------------Modified GaussSeidel-----------------------------------------------'

    gase2 = GaussSeidel(s,compute_spectral_radius=True,use_modified_method=True,warm_start=True)

    gase2.solve(tol=1e-10,max_iters=2000)

    print(gase2,end='\n\n')

    '-----------------------------------------------Milaszewicz followed by Standard Jacobi-----------------------------------------------'

    mil_sj = Milaszewicz(s,k=4,method='jacobi',use_modified_method=False,compute_spectral_radius=True,copy=True,warm_start=True)
    
    mil_sj.solve(tol=1e-6,max_iters=2000)
    
    print(mil_sj,end='\n\n')

    '-----------------------------------------------Milaszewicz followed by Modified Jacobi-----------------------------------------------'

    mil_mj = Milaszewicz(s,k=4,method='jacobi',use_modified_method=True,compute_spectral_radius=True,copy=True,warm_start=True)
    
    mil_mj.solve(tol=1e-6,max_iters=2000)
    
    print(mil_mj,end='\n\n')

