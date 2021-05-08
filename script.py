# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 23:26:04 2021

@author: Prathamesh
"""

from solve.solvers import Jacobi, Milaszewicz, GaussSeidel

from generate.generators import SystemGenerator

from visualize import visualizers as vz

if __name__ == '__main__':
    
    dim = 500
    
    kind = 'Z'
    
    sg = SystemGenerator() # system generator
    
    s = sg.generate(dim=dim,kind=kind,diagonally_dominant=True) # generate a system of linear equations
    
    method_names = []
    iteration_values = []
    spectral_radius_values = []
    
    '-----------------------------------------------Standard Jacobi-----------------------------------------------'
    
    jac1 = Jacobi(s,compute_spectral_radius=True,use_modified_method=False,warm_start=True)
    
    jac1.solve(tol=1e-10,max_iters=2000)
    
    vz.show_l_inf_plot(kind=s.kind,dim=s.dim,method='Standard Jacobi',spectral_radius=jac1.spectral_radius,l_inf_values=jac1.l_inf_values,s=2)
    
    method_names.append('Standard Jacobi')
    
    iteration_values.append(jac1.noi)
    
    spectral_radius_values.append(jac1.spectral_radius)
    
    print(jac1,end='\n\n')
    
    '-----------------------------------------------Modified Jacobi-----------------------------------------------'
    
    
    jac2 = Jacobi(s,compute_spectral_radius=True,use_modified_method=True,warm_start=True)
    
    jac2.solve(tol=1e-6,max_iters=2000)
    
    vz.show_l_inf_plot(kind=s.kind,dim=s.dim,method='Modified Jacobi',spectral_radius=jac2.spectral_radius,l_inf_values=jac2.l_inf_values,s=0.5)
    
    method_names.append('Modified Jacobi')
    
    iteration_values.append(jac2.noi)
    
    spectral_radius_values.append(jac2.spectral_radius)
    
    print(jac2,end='\n\n')
    
    '-----------------------------------------------Milaszewicz followed by Standard Jacobi-----------------------------------------------'

    mil_sj = Milaszewicz(s,k=4,method='jacobi',use_modified_method=False,compute_spectral_radius=True,copy=True,warm_start=True)
    
    mil_sj.solve(tol=1e-6,max_iters=2000)
    
    vz.show_l_inf_plot(kind=s.kind,dim=s.dim,method='Milaszewicz followed by Standard Jacobi',spectral_radius=mil_sj.solver.spectral_radius,l_inf_values=mil_sj.solver.l_inf_values,s=0.5)
    
    method_names.append('Milaszewicz followed by Standard Jacobi')
    
    iteration_values.append(mil_sj.solver.noi)
    
    spectral_radius_values.append(mil_sj.solver.spectral_radius)
    
    print(mil_sj,end='\n\n')

    '-----------------------------------------------Milaszewicz followed by Modified Jacobi-----------------------------------------------'

    mil_mj = Milaszewicz(s,k=4,method='jacobi',use_modified_method=True,compute_spectral_radius=True,copy=True,warm_start=True)
    
    mil_mj.solve(tol=1e-6,max_iters=2000)
    
    vz.show_l_inf_plot(kind=s.kind,dim=s.dim,method='Milaszewicz followed by Modified Jacobi',spectral_radius=mil_mj.solver.spectral_radius,l_inf_values=mil_mj.solver.l_inf_values,s=0.5)
    
    method_names.append('Milaszewicz followed by Modified Jacobi')
    
    iteration_values.append(mil_mj.solver.noi)
    
    spectral_radius_values.append(mil_mj.solver.spectral_radius)
    
    print(mil_mj,end='\n\n')

    '-----------------------------------------------Standard GaussSeidel-----------------------------------------------'

    gase1 = GaussSeidel(s,compute_spectral_radius=True,use_modified_method=False,warm_start=True)

    gase1.solve(tol=1e-10,max_iters=2000)
    
    vz.show_l_inf_plot(kind=s.kind,dim=s.dim,method='Standard GaussSeidel',spectral_radius=gase1.spectral_radius,l_inf_values=gase1.l_inf_values,s=0.5)
    
    method_names.append('Standard GaussSeidel')
    
    iteration_values.append(gase1.noi)
    
    spectral_radius_values.append(gase1.spectral_radius)

    print(gase1,end='\n\n')

    '-----------------------------------------------Modified GaussSeidel-----------------------------------------------'

    gase2 = GaussSeidel(s,compute_spectral_radius=True,use_modified_method=True,warm_start=True)

    gase2.solve(tol=1e-10,max_iters=2000)
    
    vz.show_l_inf_plot(kind=s.kind,dim=s.dim,method='Modified GaussSeidel',spectral_radius=gase2.spectral_radius,l_inf_values=gase2.l_inf_values,s=0.5)
    
    method_names.append('Modified GaussSeidel')
    
    iteration_values.append(gase2.noi)
    
    spectral_radius_values.append(gase2.spectral_radius)

    print(gase2,end='\n\n')

    '-----------------------------------------------Milaszewicz followed by Standard Gauss Seidel-----------------------------------------------'

    mil_sgs = Milaszewicz(s,k=4,method='gauss_seidel',use_modified_method=False,compute_spectral_radius=True,copy=True,warm_start=True)
    
    mil_sgs.solve(tol=1e-6,max_iters=2000)
    
    vz.show_l_inf_plot(kind=s.kind,dim=s.dim,method='Milaszewicz followed by Standard Gauss Seidel',spectral_radius=mil_sgs.solver.spectral_radius,l_inf_values=mil_sgs.solver.l_inf_values,s=0.5)
    
    method_names.append('Milaszewicz followed by Standard Gauss Seidel')
    
    iteration_values.append(mil_sgs.solver.noi)
    
    spectral_radius_values.append(mil_sgs.solver.spectral_radius)
    
    print(mil_sgs,end='\n\n')
    
    '-----------------------------------------------Milaszewicz followed by Modified Gauss Seidel-----------------------------------------------'

    mil_mgs = Milaszewicz(s,k=4,method='gauss_seidel',use_modified_method=True,compute_spectral_radius=True,copy=True,warm_start=True)
    
    mil_mgs.solve(tol=1e-6,max_iters=2000)
    
    vz.show_l_inf_plot(kind=s.kind,dim=s.dim,method='Milaszewicz followed by Modified Gauss Seidel',spectral_radius=mil_mgs.solver.spectral_radius,l_inf_values=mil_mgs.solver.l_inf_values,s=0.5)
    
    method_names.append('Milaszewicz followed by Modified Gauss Seidel')
    
    iteration_values.append(mil_mgs.solver.noi)
    
    spectral_radius_values.append(mil_mgs.solver.spectral_radius)
    
    print(mil_mgs,end='\n\n')

    '-----------------------------------------------Iterations and Spectral Radius Plots-----------------------------------------------'

    vz.show_iterations_plot(kind=s.kind,dim=s.dim,y=method_names,iteration_values=iteration_values)

    vz.show_spectral_radius_plot(kind=s.kind,dim=s.dim,y=method_names,spectral_radius_values=spectral_radius_values)

