# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 18:30:21 2021

@author: Prthamesh
"""

import numpy as np
    
class Jacobi:
    
    def __init__(self,A,b,x = None):
        self.A = A
        self.b = b
        self.n = A.shape[0]
        if x:
            self.x = x
        else:
            self.x = np.ones(self.n)
    
    def split(self):
        """
            Compute T and c
        """
        self.c = self.b/self.A[range(self.n),range(self.n)] # (D^-1)*b
        for i in range(self.n):
            d = self.A[i,i]
            self.A[i,:] = (-self.A[i,:])/d
            self.A[i,i] = 0
        self.T = self.A
        
    def solve(self,tol,iters):
        self.split()
        x = self.x
        for i in range(1,iters+1):
            x_new = np.dot(self.T,x) + self.c
            if np.allclose(x,x_new,atol=1e-10):
                self.x = x_new
                self.noi = i
                self.converged = True
                return
            x = x_new
        else:
            self.x = x_new
            self.noi = i
            self.converged = False
            
    def __str__(self):
        s = 'Solution :\n{}\nNumber of Iterations : {}\nConverged : {}'.format(self.x,self.noi,self.converged)
        return s

# initialize the matrix

import time

n = 10

A = (1000*np.random.randn(n,n)).astype(np.double)

for i in range(n):
    A[i,i] = np.sum(np.abs(A[range(n),range(n)])) - np.abs(A[i,i]) + np.random.randint(10,1000)
    if A[i,i] == 0:
        print(True)

x = 20*np.random.randn(n)

b = np.dot(A,x)

jac = Jacobi(A,b)

st = time.time()

jac.solve(1e-5,10000)

et = time.time()

print(et-st)

jac.converged

jac.x

jac.noi

A[:3,:3]

l = A[range(n),range(n)]


print(jac)


