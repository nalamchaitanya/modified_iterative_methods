# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 23:26:04 2021

@author: Prathamesh
"""

from solve.solvers import Jacobi

from generate.generators import SystemGenerator

if __name__ == '__main__':

    sg = SystemGenerator() # system generator
    
    s = sg.generate(dim=500,kind='positive',diagonally_dominant=True) # generate a system of linear equations
    
    # A,b = s.A,s.b
    
    '-----------------------------------------------Standard Jacobi-----------------------------------------------'
    
    jac1 = Jacobi(s,compute_spectral_radius=True,use_modified_method=False,warm_start=True)
    
    jac1.solve(tol=1e-10,max_iters=2000)
    
    print(jac1,end='\n\n')
    
    '-----------------------------------------------Modified Jacobi-----------------------------------------------'
    
    
    jac2 = Jacobi(s,compute_spectral_radius=True,use_modified_method=True,warm_start=True)
    
    jac2.solve(tol=1e-6,max_iters=2000)
    
    print(jac2,end='\n\n')

