# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 23:26:04 2021

@author: Prathamesh
"""

from solve.solvers import Jacobi, Milaszewicz, GaussSeidel

from generate.generators import SystemGenerator

from visualize import visualizers as vz

import numpy as np

if __name__ == '__main__':

    np.random.seed(0);

    dim = 500

    kind = 'Z'

    tol = 1e-6

    max_iters = 2000

    sg = SystemGenerator() # system generator
    
    s = sg.generate(dim=dim,kind=kind,diagonally_dominant=True) # generate a system of linear equations
    
    method_names = []
    iteration_values = []
    spectral_radius_values = []
    # A,b = s.A,s.b
    
    '-----------------------------------------------Standard Jacobi-----------------------------------------------'
    
    jac1 = Jacobi(s,compute_spectral_radius=True,use_modified_method=False,warm_start=True)
    
    jac1.solve(tol=tol,max_iters=max_iters)

    method_names.append('SJ')

    iteration_values.append(jac1.noi)

    spectral_radius_values.append(jac1.spectral_radius)
    
    print(jac1,end='\n\n')

    '-----------------------------------------------Modified Jacobi-----------------------------------------------'
    
    
    jac2 = Jacobi(s,compute_spectral_radius=True,use_modified_method=True,warm_start=True)
    
    jac2.solve(tol=tol,max_iters=max_iters)
    
    method_names.append('MJ '+str(jac2.diagonal_list))

    iteration_values.append(jac2.noi)

    spectral_radius_values.append(jac2.spectral_radius)

    print(jac2,end='\n\n')

    '-----------------------------------------------Modified Jacobi-----------------------------------------------'
    
    
    jac3 = Jacobi(s,compute_spectral_radius=True,use_modified_method=True,warm_start=True,diagonal_list=[1,2])
    
    jac3.solve(tol=tol,max_iters=max_iters)

    method_names.append('MJ '+str(jac3.diagonal_list))

    iteration_values.append(jac3.noi)

    spectral_radius_values.append(jac3.spectral_radius)
    
    print(jac3,end='\n\n')

    '-----------------------------------------------Modified Jacobi-----------------------------------------------'
    
    
    jac4 = Jacobi(s,compute_spectral_radius=True,use_modified_method=True,warm_start=True,diagonal_list=[1,-1])
    
    jac4.solve(tol=tol,max_iters=max_iters)

    method_names.append('MJ '+str(jac4.diagonal_list))

    iteration_values.append(jac4.noi)

    spectral_radius_values.append(jac4.spectral_radius)
    
    print(jac4,end='\n\n')

    '-----------------------------------------------Modified Jacobi-----------------------------------------------'
    
    
    jac5 = Jacobi(s,compute_spectral_radius=True,use_modified_method=True,warm_start=True,diagonal_list=[1,3])
    
    jac5.solve(tol=tol,max_iters=max_iters)

    method_names.append('MJ '+str(jac5.diagonal_list))

    iteration_values.append(jac5.noi)

    spectral_radius_values.append(jac5.spectral_radius)
    
    print(jac5,end='\n\n')

    '-----------------------------------------------Milaszewicz followed by Standard Jacobi-----------------------------------------------'

    mil_sj = Milaszewicz(s,k=4,method='jacobi',use_modified_method=False,compute_spectral_radius=True,copy=True,warm_start=True)
    
    mil_sj.solve(tol=tol,max_iters=max_iters)

    method_names.append('MilSJ')

    iteration_values.append(mil_sj.solver.noi)

    spectral_radius_values.append(mil_sj.solver.spectral_radius)
    
    print(mil_sj,end='\n\n')

    '-----------------------------------------------Milaszewicz followed by Modified Jacobi-----------------------------------------------'

    mil_mj = Milaszewicz(s,k=4,method='jacobi',use_modified_method=True,compute_spectral_radius=True,copy=True,warm_start=True)
    
    mil_mj.solve(tol=tol,max_iters=max_iters)

    method_names.append('MilMJ '+str(mil_mj.solver.diagonal_list))

    iteration_values.append(mil_mj.solver.noi)

    spectral_radius_values.append(mil_mj.solver.spectral_radius)
    
    print(mil_mj,end='\n\n')

    '-----------------------------------------------Milaszewicz followed by Multi Modified Jacobi-----------------------------------------------'

    mil_mj2 = Milaszewicz(s,k=4,method='jacobi',use_modified_method=True,compute_spectral_radius=True,copy=True,warm_start=True,diagonal_list=[1,2])
    
    mil_mj2.solve(tol=tol,max_iters=max_iters)

    method_names.append('MilMJ '+str(mil_mj2.solver.diagonal_list))

    iteration_values.append(mil_mj2.solver.noi)

    spectral_radius_values.append(mil_mj2.solver.spectral_radius)
    
    print(mil_mj2,end='\n\n')

    '-----------------------------------------------Iterations and Spectral Radius Plots-----------------------------------------------'

    vz.show_iterations_plot(kind=s.kind,dim=s.dim,y=method_names,iteration_values=iteration_values)

    vz.show_spectral_radius_plot(kind=s.kind,dim=s.dim,y=method_names,spectral_radius_values=spectral_radius_values)

