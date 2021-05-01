# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 22:55:57 2021

@author: Prathamesh
"""

import numpy as np

class MatrixGenerator:
    
    def __init__(self):
        pass
    
    def generate_random(self,dim,diagonally_dominant=False):
        
        """ generate random matrix. """
        
        A = np.random.randn(dim,dim)
        
        if diagonally_dominant:
            for r in range(dim):
                A[r,r] = np.sum(np.abs(A[r,:])) - np.abs(A[r,r]) + np.random.randint(1,10)
        
        return A
    
    def generate_nonnegative(self,dim,diagonally_dominant=False):
        
        """ generate matrix where all entries are nonnegative. """
        
        A = np.random.rand(dim,dim)
        
        if diagonally_dominant:
            for r in range(dim):
                A[r,r] = np.sum(np.abs(A[r,:])) - np.abs(A[r,r]) + np.random.randint(1,10)
        
        return A
        
    
    def generate_positive(self,dim,diagonally_dominant=False):
        
        """ generate matrix where all entries are positive. """
        
        A = np.random.rand(dim,dim)
        A = A + (np.random.rand(dim,dim) * (np.isclose(A,np.zeros((dim,dim)),atol=1e-8)).astype(np.float64))
        
        if diagonally_dominant:
            for r in range(dim):
                A[r,r] = np.sum(np.abs(A[r,:])) - np.abs(A[r,r]) + np.random.randint(1,10)
        
        return A
    
    def generate_Z(self,dim,diagonally_dominant=False):
        
        """
            generate Z matrix. 
            All non-diagonal elements ∈ (-1,0].
        """
        
        A = - np.random.rand(dim,dim)
        A[range(dim),range(dim)] = A[range(dim),range(dim)]*np.random.choice([-1,1],size=dim,replace=True)
        
        if diagonally_dominant:
            for r in range(dim):
                A[r,r] = np.sum(np.abs(A[r,:])) - np.abs(A[r,r]) + np.random.randint(1,10)
        
        return A
    
    def generate_Q(self,dim,diagonally_dominant=False):
        
        """ generate Q matrix. """
        
        A = -np.random.rand(dim,dim)
        
        A[range(dim),range(dim)] = 0
        
        for c in range(dim):
            A[c,c] = - np.sum(A[:,c])
        
        return A
    
    def generate_tridiagonal(self,dim,diagonally_dominant):
        
        """ generate tridiagonal matrix. """
        
        A = np.zeros((dim,dim),dtype=np.float64)
        A[range(1,dim),range(0,dim-1)] = np.random.randn(dim-1)
        A[range(0,dim-1),range(1,dim)] = np.random.randn(dim-1)
        
        if diagonally_dominant:
            for r in range(dim):
                if r == 0:
                    A[r,r] = np.abs(A[r,1]) + np.random.randint(1,10)
                elif r == (dim-1):
                    A[r,r] = np.abs(A[r,r-1]) + np.random.randint(1,10)
                else:
                    A[r,r] = np.abs(A[r,r-1]) + np.abs(A[r,r+1]) + np.random.randint(1,10)
        
        return A
    
    def generate_triangular(self,dim,diagonally_dominant,upper):
        
        """ generate triangular matrix. """
        
        A = np.random.randn(dim,dim)
        if upper:
            for r in range(1,dim):
                A[r,:r] = 0
        else:
            for r in range(dim-1):
                A[r,r+1:] = 0
        
        if diagonally_dominant:
            for r in range(dim):
                A[r,r] = np.sum(np.abs(A[r,:])) - np.abs(A[r,r]) + np.random.randint(1,10)
        
        return A
    
    def generate(self,dim=10,kind='random',diagonally_dominant=False):
        if kind == 'random':
            return self.generate_random(dim,diagonally_dominant)
        elif kind == 'nonnegative':
            return self.generate_nonnegative(dim,diagonally_dominant)
        elif kind == 'positive':
            return self.generate_positive(dim,diagonally_dominant)
        elif kind == 'Z':
            return self.generate_Z(dim,diagonally_dominant)
        elif kind == 'Q':
            return self.generate_Q(dim)
        elif kind == 'tridiagonal':
            return self.generate_tridiagonal(dim,diagonally_dominant)
        elif kind == 'triangular_U' or kind == 'triangular_L':
            return self.generate_triangular(dim,diagonally_dominant,kind.split('_')[1]=='U')
        
class System:
    
    def __init__(self,A,x_true,b,kind,diagonally_dominant,dim):
        self.A = A
        self.x_true = x_true
        self.b = b
        self.kind = kind
        self.diagonally_dominant = diagonally_dominant
        self.dim = dim
        
class SystemGenerator:
    
    def __init__(self):
        self.mg = MatrixGenerator()
    
    def generate(self,dim=10,kind='Z',diagonally_dominant=False):
        
        """
            generate a system of equations Ax = b.
            returns system object.
        """
        
        A = self.mg.generate(dim,kind,diagonally_dominant)
        x_true = np.random.rand(dim)
        b = np.dot(A,x_true)
        return System(A,x_true,b,kind,diagonally_dominant,dim)


