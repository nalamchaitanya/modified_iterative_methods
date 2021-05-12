# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 22:55:57 2021

@author: Prathamesh

"""

import numpy as np
from data.data import loadMatrix

class MatrixGenerator:

    """ Class MatrixGenerator helps in generating system of equations with the matrices satisfying different conditions
        for example diagonal dominance, positive, non-negative, symmetric, Z, Q, triangular, tridiagonal etc. Matrices
        are initially generated using numpy random and then modified to satisfy the properties.
    """
    
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

        """ Single function encompassing that calls other functions based on the 'kind' of matrix needed to be generated"""
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

    """ Class System is a place holder class for encompassing all the things surrounding a system of equations that are
        useful for conducting experiments by applying different iterative methods. Main components of the linear system of equations
        Ax = b are:

        * A - Matrix of the system of equations Ax = b and we only stick to square matrices for the purpose of this project
        * x_true - True solution for the system of equations Ax = b
        * b - This is the product of A and x_true which ensures that the system is consistent with a unique solution
        * kind - Kind of the matrix A that is generated by random method or by loading from Matrix Market
        * diagonally_dominant - Whether the matrix A is diagonally dominant or not
        * dim - Dimension of the vectore x, b, A as square is assumed

    """
    
    def __init__(self,A,x_true,b,kind,diagonally_dominant,dim):
        self.A = A
        self.x_true = x_true
        self.b = b
        self.kind = kind
        self.diagonally_dominant = diagonally_dominant
        self.dim = dim
        
class SystemGenerator:

    """ Class SystemGenerator uses MatrixGenerator class to create a System class which acts as a place holder object
        containing the system of equations we use as input for applying different iterative methods
    """
    
    def __init__(self):

        """ Consists only a MatrixGenerator as its member variable which is used to generate any number of system of equations
            whenever the method generate is called.
        """
        self.mg = MatrixGenerator()

    def load(self, file, kind, diagonally_dominant):

        """ loads the matrix from the given input file of format .mtx and uses it to generate a system of equations and thus
            creating the required System object which describes the system of equations for the purpose of applying iterative
            methods
        """
        A = loadMatrix(file)
        x_true = np.random.rand(A.shape[0])
        b = np.dot(A,x_true)
        return System(A,x_true,b,kind,diagonally_dominant,A.shape[0])
    
    def generate(self,dim=10,kind='Z',diagonally_dominant=False):
        
        """
            generate a system of equations Ax = b.
            returns system object.
        """
        
        A = self.mg.generate(dim,kind,diagonally_dominant)
        x_true = np.random.rand(dim)
        b = np.dot(A,x_true)
        return System(A,x_true,b,kind,diagonally_dominant,dim)


