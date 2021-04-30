# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 18:21:05 2021

@author: Prathamesh
"""

from tabulate import tabulate
import numpy as np
from numpy.linalg import eigvals, inv
    
class Jacobi:
    
    """Standard and modified jacobi methods."""
    
    def __init__(self,system,x = None,use_modified_method=False,compute_spectral_radius=False,copy=True,warm_start=False):
        
        if copy:
            self.A = system.A.copy() # square matrix A
            self.b = system.b.copy() # vector b
        else:
            self.A = system.A # square matrix A
            self.b = system.b # vector b
            
        self.n = self.A.shape[0] # dimension of a square matrix A
        self.kind = system.kind # kind of matrix A
        if system.diagonally_dominant: self.kind += ', diagonally dominant'
        self._use_modified_method = use_modified_method # flag to indicate the use of modified method
        self._compute_spectral_radius = compute_spectral_radius # flag to indicate whether to compute spectral radius of iteration matrix
        self.warm_start = warm_start # flag to indicate whether to warm start
        
        self.noi = 0 # initialize number of iterations
        if x: self.x = x # initial solution vector is given
        else: self.x = np.ones(self.n) # initial solution vector is a vector of ones
        
        if not use_modified_method:
            if self.__zero_along_diagonal(): # diagonal of A contains 0
                self._zero_check_passed = False # flag to indicate that the zero check failed
                print('Diagonal of the matrix cannot contain 0') # message about the failure to the user
            else:
                self._zero_check_passed = True # flag to indicate that the zero check passed
        else:
            self._zero_check_passed = True
            
        self.splitted = False # flag to indicate the status of splitting
             
    def __zero_along_diagonal(self):
        
        """Check if there is a zero along the diagonal."""
        
        if (np.isclose(self.A[range(self.n),range(self.n)],np.zeros(self.n),atol=1e-9)).any(): # diagonal of A contains 0
            return True
        else: # diagonal of A does not contain 0
            return False
    
    def __all_ones_along_diagonal(self):
        
        """Check if all the diagonal elements are one."""
        
        if (self.A[range(self.n),range(self.n)] == np.ones(self.n)).all(): # all elements along the diagonal of A are 1
            return True
        else: # some element along the diagonal of A is not 1
            return False
        
    def compute_spectral_radius(self):
        
        """Compute spectral radius of iteration matrix."""
        
        self.spectral_radius = np.max(np.abs(eigvals(self.T))) # compute spectral radius
    
    def split(self):
        
        """Compute matrix T and vector c in the iteration x(k+1) = T * x(k) + c"""
        
        if not self._use_modified_method: # standard jacobi method
            self.c = self.b/self.A[range(self.n),range(self.n)] # (D^-1)*b
            self.T = -1 * np.divide(self.A,self.A[range(self.n),range(self.n)].reshape(self.n,1)) # iteration matrix T
            self.T[range(self.n),range(self.n)] = 0
        else: # modified jacobi method
            
            if not self.__all_ones_along_diagonal():
                self.b = self.b / self.A[range(self.n),range(self.n)] # (D^-1)*b
                self.A = np.divide(self.A,self.A[range(self.n),range(self.n)].reshape(self.n,1)) # (D^-1)*A
        
            S = np.zeros((self.n,self.n),dtype=np.float64)
            S[range(0,self.n-1),range(1,self.n)] = -self.A[range(0,self.n-1),range(1,self.n)] # matrix S
            
            L = np.zeros((self.n,self.n),dtype=np.float64) # strictly lower triangular matrix
            for r in range(1,self.n):
                L[r,0:r] = - self.A[r,0:r]
            
            U = np.zeros((self.n,self.n),dtype=np.float64) # strictly upper triangular matrix
            for r in range(0,self.n-1):
                U[r,r+1:] = - self.A[r,r+1:]
            
            I_SL_inv = inv(np.eye(self.n)-np.matmul(S,L)) # matrix (I - SL)^-1
            
            self.c = np.dot(I_SL_inv,np.dot(np.eye(self.n)+S,self.b))
            self.T = np.matmul(I_SL_inv,L+U-S+np.matmul(S,U)) # iteration matrix T
                    
        if self._compute_spectral_radius: # compute spectral radius of iteration matrix T
            self.compute_spectral_radius()

# TODO move iterate, solve to utility class so that it can be modular as well as good to time only the iteration procedure.
    def __iterate(self,tol,max_iters):
        
        """Perform iterations."""
        
        x = self.x # initial solution vector
        
        for i in range(1,max_iters+1):
            x_new = np.dot(self.T,x) + self.c # compute new solution vector
            if np.allclose(self.b,np.dot(self.A,x_new),atol=tol): # if old solution vector and new solution vector are almost equal
                self.x = x_new # final solution vector
                self.noi += i # number of iterations taken
                self.converged = True # flag to indicate convergence
                return
            x = x_new
        else:
            self.x = x_new # final solution vector
            self.noi += i # number of iterations taken
            self.converged = False # flag to indicate convergence
        
    def solve(self,tol=1e-5,max_iters=100):
        
        """Compute the solution vector for a given system."""
        
        if not self.warm_start:
            self.x = np.ones()
            self.noi = 0
        
        if not self._zero_check_passed: # zero along diagonal
            print('System cannot be solved as there is a 0 along the diagonal of a matrix.')
            return
            
        if not self.splitted: # matrix T and vector c are not yet computed
            self.split() # compute matrix T and vector c
            self.splitted = True
        
        self.__iterate(tol,max_iters) # perform iterations
            
    def __str__(self):
        
        method = 'Modified Jacobi' if self._use_modified_method else 'Standard Jacobi'
        
        table = [['Kind',self.kind],
                 ['Dimension',self.n],
                 ['Method',method],
                 ['# Iterations',self.noi],
                 ['Converged',self.converged]]
        
        if self.compute_spectral_radius:
            table.append(['Spectral radius',self.spectral_radius])
        
        return tabulate(table,tablefmt='grid')

class Milaszewicz:
    
    def __init__(self,system,k,method='jacobi',x = None,use_modified_method=False,compute_spectral_radius=False,copy=True,warm_start=False):
        
        self.k = k
        self.method = method
        
        if method == 'jacobi':
            self.solver = Jacobi(system,x,use_modified_method,compute_spectral_radius,copy,warm_start)
            self.solver_name = 'Jacobi'
        elif method == 'gauss_seidel':
            self.solver = Jacobi(system,x,use_modified_method,compute_spectral_radius,copy,warm_start)
            self.solver_name = 'GaussSeidel'
        
        self.n = self.solver.n
        
        self._elimination_performed = False
    
    def perform_elimination(self):
        
        """
            x(k+1) = T * x(k) + c
            is transformed to
            x(k+1) = (S*T + T - S) + (I + S)*c.
        """
        
        T = self.solver.T # iteration matrix T of the solver
        c = self.solver.c 
        
        S = np.zeros((self.n,self.n))
        S[:,self.k] = T[:,self.k]
        
        T = np.matmul(S,T) + T - S
        c = np.dot(np.eye(self.n)+S,c)
        
        self.solver.T = T # new iteration matrix T of the solver
        self.solver.c = c
        
        if self.solver._compute_spectral_radius: # compute spectral radius of new iteration matrix T
            self.solver.compute_spectral_radius()
    
    def solve(self,tol=1e-5,max_iters=100):
        
        """Compute the solution vector for a given system."""
        
        if not self._elimination_performed: # elimination not performed
            self.solver.split() # compute matrix T and vector c of the solver
            self.solver.splitted = True
            self.perform_elimination() # perform elimination
            self._elimination_performed = True
        
        self.solver.solve(tol,max_iters) # perform iterations and compute solution vector
        
        self.x = self.solver.x # solution vector
    
    def __str__(self):
       
        solver_name = self.solver_name
        
        method = f'Milaszewicz followed by Modified {solver_name}' if self.solver._use_modified_method else f'Milaszewicz followed by Standard {solver_name}'
        
        table = [['Kind',self.solver.kind],
                 ['Dimension',self.solver.n],
                 ['Method',method],
                 ['# Iterations',self.solver.noi],
                 ['Converged',self.solver.converged]]
        
        if self.solver._compute_spectral_radius:
            table.append(['Spectral radius',self.solver.spectral_radius])
        
        return tabulate(table,tablefmt='grid')

class GaussSeidel:

    """GaussSeidel iterative method"""

    def __init__(self,system,x = None,use_modified_method=False,compute_spectral_radius=False,copy=True,warm_start=False):

        if copy:
            self.A = system.A.copy() # square matrix A
            self.b = system.b.copy() # vector b
        else:
            self.A = system.A # square matrix A
            self.b = system.b # vector b
        
        self.n = self.A.shape[0] # dimension of a square matrix A
        self.kind = system.kind # kind of matrix A
        if system.diagonally_dominant: self.kind += ', diagonally dominant'
        self._use_modified_method = use_modified_method # flag to indicate the use of modified method
        self._compute_spectral_radius = compute_spectral_radius # flag to indicate whether to compute spectral radius of iteration matrix
        self.warm_start = warm_start # flag to indicate whether to warm start
        
        self.noi = 0 # initialize number of iterations
        if x: self.x = x # initial solution vector is given
        else: self.x = np.ones(self.n) # initial solution vector is a vector of ones

        if not use_modified_method:
            if self.__zero_along_diagonal(): # diagonal of A contains 0
                self._zero_check_passed = False # flag to indicate that the zero check failed
                print('Diagonal of the matrix cannot contain 0') # message about the failure to the user
            else:
                self._zero_check_passed = True # flag to indicate that the zero check passed
        else:
            self._zero_check_passed = True
            
        self.splitted = False # flag to indicate the status of splitting
             
    # TODO Move these functions to utility class.
    def __zero_along_diagonal(self):
        
        """Check if there is a zero along the diagonal."""
        
        if (np.isclose(self.A[range(self.n),range(self.n)],np.zeros(self.n),atol=1e-9)).any(): # diagonal of A contains 0
            return True
        else: # diagonal of A does not contain 0
            return False
    
    def __all_ones_along_diagonal(self):
        
        """Check if all the diagonal elements are one."""
        
        if (self.A[range(self.n),range(self.n)] == np.ones(self.n)).all(): # all elements along the diagonal of A are 1
            return True
        else: # some element along the diagonal of A is not 1
            return False
        
    def compute_spectral_radius(self):
        
        """Compute spectral radius of iteration matrix."""
        
        self.spectral_radius = np.max(np.abs(eigvals(self.T))) # compute spectral radius

    def split(self):
        
        """Compute matrix T and vector c in the iteration x(k+1) = T * x(k) + c"""

        if not self._use_modified_method: # standard gauss-seidel method
            Ls = np.zeros((self.n, self.n));
            for r in range(self.n):
                Ls[r,:r+1] = self.A[r,:r+1];

            U = np.zeros((self.n,self.n));
            for r in range(self.n):
                U[r,r+1:] = self.A[r,r+1:];

            # TODO Can be optimized by finding forward substitution way of finding inverse
            Lsi = inv(Ls);
            self.c = np.dot(Lsi,self.b);
            self.T = - np.matmul(Lsi,U);

        else: #modified gauss-seidel method
            if not self.__all_ones_along_diagonal():
                self.b = self.b / self.A[range(self.n),range(self.n)];
                self.A = np.divide(self.A, self.A[range(self.n),range(self.n)].reshape(self.n,1));

            S = np.zeros((self.n,self.n), dtype = np.float64);
            S[range(0,self.n-1),range(1,self.n)] = -self.A[range(0,self.n-1),range(1,self.n)];

            L = np.zeros((self.n,self.n),dtype=np.float64) # strictly lower triangular matrix
            for r in range(1,self.n):
                L[r,0:r] = - self.A[r,0:r]
            
            U = np.zeros((self.n,self.n),dtype=np.float64) # strictly upper triangular matrix
            for r in range(0,self.n-1):
                U[r,r+1:] = - self.A[r,r+1:]

            # TODO condition for existence of inverse.
            I_L_SL_inv = inv(np.eye(self.n) - L - np.matmul(S,L));

            self.c = np.dot(I_L_SL_inv, np.dot(np.eye(self.n)+S,self.b));
            self.T = np.matmul(I_L_SL_inv, np.matmul(S,U) + U - S);
            
        if self._compute_spectral_radius: # compute spectral radius of iteration matrix T
            self.compute_spectral_radius()


    # TODO move iterate, solve to utility class so that it can be modular as well as good to time only the iteration procedure.
    def __iterate(self,tol,max_iters):
        
        """Perform iterations."""
        
        x = self.x # initial solution vector
        
        for i in range(1,max_iters+1):
            x_new = np.dot(self.T,x) + self.c # compute new solution vector
            if np.allclose(self.b,np.dot(self.A,x_new),atol=tol): # if old solution vector and new solution vector are almost equal
                self.x = x_new # final solution vector
                self.noi += i # number of iterations taken
                self.converged = True # flag to indicate convergence
                return
            x = x_new
        else:
            self.x = x_new # final solution vector
            self.noi += i # number of iterations taken
            self.converged = False # flag to indicate convergence
        
    def solve(self,tol=1e-5,max_iters=100):
        
        """Compute the solution vector for a given system."""
        
        if not self.warm_start:
            self.x = np.ones()
            self.noi = 0
        
        if not self._zero_check_passed: # zero along diagonal
            print('System cannot be solved as there is a 0 along the diagonal of a matrix.')
            return
            
        if not self.splitted: # matrix T and vector c are not yet computed
            self.split() # compute matrix T and vector c
            self.splitted = True
        
        self.__iterate(tol,max_iters) # perform iterations
            
    def __str__(self):
        
        method = 'Modified Gauss-Seidel' if self._use_modified_method else 'Standard Gauss-Seidel'
        
        table = [['Kind',self.kind],
                 ['Dimension',self.n],
                 ['Method',method],
                 ['# Iterations',self.noi],
                 ['Converged',self.converged]]
        
        if self.compute_spectral_radius:
            table.append(['Spectral radius',self.spectral_radius])
        
        return tabulate(table,tablefmt='grid')



