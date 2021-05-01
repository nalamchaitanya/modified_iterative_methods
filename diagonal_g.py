# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 23:26:04 2021

@author: Prathamesh
"""

from solve.solvers import Jacobi, Milaszewicz, GaussSeidel

from generate.generators import SystemGenerator

if __name__ == '__main__':

    sg = SystemGenerator() # system generator
    
    s = sg.generate(dim=100,kind='Q',diagonally_dominant=True) # generate a system of linear equations
    
    # A,b = s.A,s.b
    
    '-----------------------------------------------Standard GaussSeidel-----------------------------------------------'
    
    jac1 = GaussSeidel(s,compute_spectral_radius=True,use_modified_method=False,warm_start=True)
    
    jac1.solve(tol=1e-10,max_iters=2000)
    
    print(jac1,end='\n\n')

    '-----------------------------------------------Modified GaussSeidel-----------------------------------------------'
    
    
    jac2 = GaussSeidel(s,compute_spectral_radius=True,use_modified_method=True,warm_start=True)
    
    jac2.solve(tol=1e-6,max_iters=2000)
    
    print(jac2,end='\n\n')

    '-----------------------------------------------Modified GaussSeidel-----------------------------------------------'
    
    
    jac3 = GaussSeidel(s,compute_spectral_radius=True,use_modified_method=True,warm_start=True,diagonal_list=[1,2])
    
    jac3.solve(tol=1e-6,max_iters=2000)
    
    print(jac3,end='\n\n')

    '-----------------------------------------------Modified GaussSeidel-----------------------------------------------'
    
    
    jac4 = GaussSeidel(s,compute_spectral_radius=True,use_modified_method=True,warm_start=True,diagonal_list=[1,-1])
    
    jac4.solve(tol=1e-6,max_iters=2000)
    
    print(jac4,end='\n\n')

    '-----------------------------------------------Milaszewicz followed by Standard GaussSeidel-----------------------------------------------'

    mil_sj = Milaszewicz(s,k=4,method='gauss_seidel',use_modified_method=False,compute_spectral_radius=True,copy=True,warm_start=True)
    
    mil_sj.solve(tol=1e-6,max_iters=2000)
    
    print(mil_sj,end='\n\n')

    