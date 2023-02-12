# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 23:26:04 2021

@author: Prathamesh
"""

from solve.solvers import Jacobi, Milaszewicz, GaussSeidel

from generate.generators import SystemGenerator

from visualize import visualizers as vz

import os
import time
import numpy as np

if __name__ == '__main__':

    np.random.seed(int(time.time()))
    
    dim = 1000
    
    kind = 'Z'

    tol = 1e-6

    max_iters = 2000

    folder = 'test'
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    sg = SystemGenerator() # system generator
    
    s = sg.generate(dim=dim,kind=kind,diagonally_dominant=True) # generate a system of linear equations
    
    method_names = []
    iteration_values = []
    spectral_radius_values = []
    
    
    '-----------------------------------------------Standard GaussSeidel-----------------------------------------------'

    gase1 = GaussSeidel(s,compute_spectral_radius=True,use_modified_method=False,warm_start=True)

    gase1.solve(tol=tol,max_iters=max_iters)
    
    # vz.show_l_inf_plot(kind=s.kind,dim=s.dim,method='Standard GaussSeidel',spectral_radius=gase1.spectral_radius,l_inf_values=gase1.l_inf_values,s=0.5,folder=folder)
    
    method_names.append(-1)
    
    iteration_values.append(gase1.noi)
    
    spectral_radius_values.append(gase1.spectral_radius)

    print(gase1,end='\n\n')

    '-----------------------------------------------Milaszewicz followed by Standard Gauss Seidel-----------------------------------------------'
    mil_sgs = Milaszewicz(s,k=int(0.9*dim),method='gauss_seidel',use_modified_method=False,compute_spectral_radius=True,copy=True,warm_start=True)
    
    mil_sgs.solve(tol=tol,max_iters=max_iters)
    
    # vz.show_l_inf_plot(kind=s.kind,dim=s.dim,method='Milaszewicz followed by Standard Gauss Seidel',spectral_radius=mil_sgs.solver.spectral_radius,l_inf_values=mil_sgs.solver.l_inf_values,s=0.5,folder=folder)
    
    method_names.append(k)
    
    iteration_values.append(mil_sgs.solver.noi)
    
    spectral_radius_values.append(mil_sgs.solver.spectral_radius)
    
    print(mil_sgs,end='\n\n')
    
    '-----------------------------------------------Iterations and Spectral Radius Plots-----------------------------------------------'

    # vz.show_iterations_plot(kind=s.kind,dim=s.dim,y=method_names,iteration_values=iteration_values,folder=folder)

    # vz.show_spectral_radius_plot(kind=s.kind,dim=s.dim,y=method_names,spectral_radius_values=spectral_radius_values,folder=folder)

