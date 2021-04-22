# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 18:21:05 2021

@author: Prathamesh
"""

import numpy as np
from numpy.linalg import eigvals, inv
    
class Jacobi:
    
    """Standard and modified jacobi methods."""
    
    def __init__(self,A,b,x = None,use_modified_method=False,compute_spectral_radius=False,copy=True):
        
        if copy:
            self.A = A.copy() # square matrix A
            self.b = b.copy() # vector b
        else:
            self.A = A # square matrix A
            self.b = b # vector b
            
        self.n = A.shape[0] # dimension of a square matrix A
        self.use_modified_method = use_modified_method # flag to indicate the use of modified method
        self.compute_spectral_radius = compute_spectral_radius # flag to indicate whether to compute spectral radius of iteration matrix
        
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
        
    def __compute_spectral_radius(self):
        
        """Compute spectral radius of iteration matrix."""
        
        self.spectral_radius = np.max(np.abs(eigvals(self.T))) # compute spectral radius
    
    def __split(self):
        
        """Compute matrix T and vector c in the iteration x(k+1) = T * x(k) + c"""
        
        if not self.use_modified_method: # standard jacobi method
            self.c = self.b/self.A[range(self.n),range(self.n)] # (D^-1)*b
            self.T = -1 * np.divide(self.A,self.A[range(self.n),range(self.n)].reshape(self.n,1)) # iteration matrix T
            self.T[range(self.n),range(self.n)] = 0
        else: # modified jacobi method
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
                    
        if self.compute_spectral_radius: # compute spectral radius of iteration matrix T
            self.__compute_spectral_radius()
        
    def __iterate(self,tol,max_iters):
        
        """Perform iterations."""
        
        x = self.x # initial solution vector
        
        for i in range(1,max_iters+1):
            x_new = np.dot(self.T,x) + self.c # compute new solution vector
            if np.allclose(self.b,np.dot(self.A,x_new),atol=tol): # if old solution vector and new solution vector are almost equal
                self.x = x_new # final solution vector
                self.noi = i # number of iterations taken
                self.converged = True # flag to indicate convergence
                return
            x = x_new
        else:
            self.x = x_new # final solution vector
            self.noi = i # number of iterations taken
            self.converged = False # flag to indicate convergence
        
    def solve(self,tol,max_iters):
        
        """Compute the solution vector for a given system."""
        
        if not self._zero_check_passed: # zero along diagonal
            print('System cannot be solved as there is a 0 along the diagonal of a matrix.')
            return
            
        self.__split() # compute matrix T and vector c
        
        self.__iterate(tol,max_iters) # perform iterations
        
            
    def __str__(self):
        if self.compute_spectral_radius:
            s = 'Solution :\n{}\nNumber of Iterations : {}\nConverged : {}\nSpectral Radius : {}'.format(self.x,self.noi,self.converged,self.spectral_radius)
        else:
            s = 'Solution :\n{}\nNumber of Iterations : {}\nConverged : {}'.format(self.x,self.noi,self.converged)
        return s


