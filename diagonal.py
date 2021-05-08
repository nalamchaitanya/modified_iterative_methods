# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 23:26:04 2021

@author: Prathamesh
"""

from solve.solvers import Jacobi, Milaszewicz, GaussSeidel

from generate.generators import SystemGenerator
import numpy as np

if __name__ == '__main__':

    np.random.seed(0);

    sg = SystemGenerator() # system generator
    
    s = sg.generate(dim=500,kind='Z',diagonally_dominant=True) # generate a system of linear equations
    
    # A,b = s.A,s.b
    
    '-----------------------------------------------Standard Jacobi-----------------------------------------------'
    
    jac1 = Jacobi(s,compute_spectral_radius=True,use_modified_method=False,warm_start=True)
    
    jac1.solve(tol=1e-10,max_iters=4000)
    
    print(jac1,end='\n\n')

    '-----------------------------------------------Modified Jacobi-----------------------------------------------'
    
    
    jac2 = Jacobi(s,compute_spectral_radius=True,use_modified_method=True,warm_start=True)
    
    jac2.solve(tol=1e-6,max_iters=4000)
    
    print(jac2,end='\n\n')

    '-----------------------------------------------Modified Jacobi-----------------------------------------------'
    
    
    jac3 = Jacobi(s,compute_spectral_radius=True,use_modified_method=True,warm_start=True,diagonal_list=[1,2])
    
    jac3.solve(tol=1e-6,max_iters=4000)
    
    print(jac3,end='\n\n')

    '-----------------------------------------------Modified Jacobi-----------------------------------------------'
    
    
    jac4 = Jacobi(s,compute_spectral_radius=True,use_modified_method=True,warm_start=True,diagonal_list=[1,3])
    
    jac4.solve(tol=1e-6,max_iters=4000)
    
    print(jac4,end='\n\n')

    '-----------------------------------------------Modified Jacobi-----------------------------------------------'
    
    
    jac4 = Jacobi(s,compute_spectral_radius=True,use_modified_method=True,warm_start=True,diagonal_list=[1,4])
    
    jac4.solve(tol=1e-6,max_iters=4000)
    
    print(jac4,end='\n\n')

    '-----------------------------------------------Modified Jacobi-----------------------------------------------'
    
    
    jac4 = Jacobi(s,compute_spectral_radius=True,use_modified_method=True,warm_start=True,diagonal_list=[1,5])
    
    jac4.solve(tol=1e-6,max_iters=4000)
    
    print(jac4,end='\n\n')

    '-----------------------------------------------Milaszewicz followed by Standard Jacobi-----------------------------------------------'

    mil_sj = Milaszewicz(s,k=4,method='jacobi',use_modified_method=True,compute_spectral_radius=True,copy=True,warm_start=True,diagonal_list=[1,2])
    
    mil_sj.solve(tol=1e-6,max_iters=4000)
    
    print(mil_sj,end='\n\n')

    