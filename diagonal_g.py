# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 23:26:04 2021

@author: Prathamesh

This file tests different modifications on top of Gauss Seidel method experimenting with making different diagonals zero and
plotting their effect on number of iterations for the given tolerance and spectral radius values of the iteration matrix.

Most of the code is similar to diagonal.py hence refer that code for more comments
"""

from solve.solvers import Jacobi, Milaszewicz, GaussSeidel

from generate.generators import SystemGenerator

from visualize import visualizers as vz

import numpy as np
import time

if __name__ == '__main__':

    """ Important Parameters used for testing the iterative methods """

    np.random.seed(int(time.time()))

    dim = 500

    kind = 'Z'

    tol = 1e-6

    max_iters = 2000

    sg = SystemGenerator() # system generator
    
    s = sg.generate(dim=dim,kind=kind,diagonally_dominant=True) # generate a system of linear equations
    
    # Arrays to store the values and experiment results to finally plot using visualizer

    method_names = []
    iteration_values = []
    spectral_radius_values = []
    # A,b = s.A,s.b
    
    '-----------------------------------------------Standard GaussSeidel-----------------------------------------------'
    
    jac1 = GaussSeidel(s,compute_spectral_radius=True,use_modified_method=False,warm_start=True)
    
    jac1.solve(tol=tol,max_iters=max_iters)

    method_names.append('Standard GaussSeidel')

    iteration_values.append(jac1.noi)

    spectral_radius_values.append(jac1.spectral_radius)
    
    print(jac1,end='\n\n')

    '-----------------------------------------------Modified GaussSeidel-----------------------------------------------'
    
    # When use_modified_method=True default value for diagonal_list is [1], a list containing only the first upper co-diagonal
    jac2 = GaussSeidel(s,compute_spectral_radius=True,use_modified_method=True,warm_start=True)
    
    jac2.solve(tol=tol,max_iters=max_iters)
    
    method_names.append('Modified GaussSeidel '+str(jac2.diagonal_list))

    iteration_values.append(jac2.noi)

    spectral_radius_values.append(jac2.spectral_radius)

    print(jac2,end='\n\n')

    '-----------------------------------------------Modified GaussSeidel-----------------------------------------------'
    
    # The default diagonal_list which determines what diagonals to be nullified by transformations
    # can be overridden by passing different value as given below, for example diagonal_list=[1,2]
    jac3 = GaussSeidel(s,compute_spectral_radius=True,use_modified_method=True,warm_start=True,diagonal_list=[1,2])
    
    jac3.solve(tol=tol,max_iters=max_iters)

    method_names.append('Modified GaussSeidel '+str(jac3.diagonal_list))

    iteration_values.append(jac3.noi)

    spectral_radius_values.append(jac3.spectral_radius)
    
    print(jac3,end='\n\n')

    '-----------------------------------------------Modified GaussSeidel-----------------------------------------------'

    jac4 = GaussSeidel(s,compute_spectral_radius=True,use_modified_method=True,warm_start=True,diagonal_list=[1,3])

    jac4.solve(tol=tol,max_iters=max_iters)

    method_names.append('Modified GaussSeidel '+str(jac4.diagonal_list))

    iteration_values.append(jac4.noi)

    spectral_radius_values.append(jac4.spectral_radius)

    print(jac4,end='\n\n')

    '-----------------------------------------------Modified GaussSeidel-----------------------------------------------'

    jac5 = GaussSeidel(s,compute_spectral_radius=True,use_modified_method=True,warm_start=True,diagonal_list=[1,-1])

    jac5.solve(tol=tol,max_iters=max_iters)

    method_names.append('Modified GaussSeidel '+str(jac5.diagonal_list))

    iteration_values.append(jac5.noi)

    spectral_radius_values.append(jac5.spectral_radius)

    print(jac5,end='\n\n')

    '-----------------------------------------------Milaszewicz followed by Standard GaussSeidel-----------------------------------------------'

    mil_sj = Milaszewicz(s,k=4,method='gauss_seidel',use_modified_method=False,compute_spectral_radius=True,copy=True,warm_start=True)

    mil_sj.solve(tol=tol,max_iters=max_iters)

    method_names.append('Milaszewicz followed by Standard GaussSeidel')

    iteration_values.append(mil_sj.solver.noi)

    spectral_radius_values.append(mil_sj.solver.spectral_radius)

    print(mil_sj,end='\n\n')

    '-----------------------------------------------Milaszewicz followed by Modified GaussSeidel-----------------------------------------------'

    mil_mj = Milaszewicz(s,k=4,method='gauss_seidel',use_modified_method=True,compute_spectral_radius=True,copy=True,warm_start=True)

    mil_mj.solve(tol=tol,max_iters=max_iters)

    method_names.append('Milaszewicz followed by Modified GaussSeidel '+str(mil_mj.solver.diagonal_list))

    iteration_values.append(mil_mj.solver.noi)

    spectral_radius_values.append(mil_mj.solver.spectral_radius)

    print(mil_mj,end='\n\n')

    '-----------------------------------------------Milaszewicz followed by Multi Modified GaussSeidel-----------------------------------------------'

    mil_mj2 = Milaszewicz(s,k=4,method='gauss_seidel',use_modified_method=True,compute_spectral_radius=True,copy=True,warm_start=True,diagonal_list=[1,2])

    mil_mj2.solve(tol=tol,max_iters=max_iters)

    method_names.append('Milaszewicz followed by Modified GaussSeidel '+str(mil_mj2.solver.diagonal_list))

    iteration_values.append(mil_mj2.solver.noi)

    spectral_radius_values.append(mil_mj2.solver.spectral_radius)

    print(mil_mj2,end='\n\n')

    '-----------------------------------------------Iterations and Spectral Radius Plots-----------------------------------------------'

    # Plots the number of iterations, spectral radius values for different iterative methods tested above

    vz.show_iterations_plot(kind=s.kind,dim=s.dim,y=method_names,iteration_values=iteration_values)

    vz.show_spectral_radius_plot(kind=s.kind,dim=s.dim,y=method_names,spectral_radius_values=spectral_radius_values)
