# -*- coding:utf-8 -*-
"""
Created on 6. feb. 2011

@author: pab
"""

# import a tool to use / as a symbol for normal division
from __future__ import division

#import system data
import sys, os

#Loading the required packages
import scipy as sp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import numdifftools as nd

# and subpackages
from scipy import *
from scipy import linalg, optimize, constants

#-----------------------------------------------------------------------------------------
#       Hamiltonian      H=sum_i(p_i2/(2m)+ 1/2 * m * w2 x_i2)+ sum_(i!=j)(a/|x_i-x_j|)
#-----------------------------------------------------------------------------------------

class classicalHamiltonian:
    def __init__(self):
       
        self.N = 2                            #N is a scalar, it's the number of ions in the chain
        f = 1000000                            #f is a scalar, it's the trap frequency
        self.w = 2*pi*f                         #w is a scalar, it's the angular velocity corresponding to the trap frequency
        self.C = (4*pi*constants.epsilon_0)**(-1)*constants.e**2    #C is a scalar, it's the Coulomb constant times the electronic charge in SI
        self.m = 39.96*1.66e-27                        #m is the mass of a single trapped ion in the chain
       
       


    def potential(self, positionvector):                     #Defines the potential that is going to be minimized
       
        x= positionvector                         #x is an 1-d array (vector) of lenght N that contains the positions of the N ions
        w= self.w
        C= self.C
        m= self.m
       
       
        #First we consider the potential of the harmonic osszilator
        Vx = 0.5 * m * (w**2) * sum(x**2)               
       
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                Vx += C /(abs(x[i]-x[j]))    #then we add the coulomb interaction
               
        return Vx
    
    def initialposition(self):        #Defines the initial position as an estimate for the minimize process
       
        N= self.N
        x_0 = r_[-(N-1)/2:(N-1)/2:N*1j]
        return x_0
       
    def normal_modes(self, eigenvalues):    #the computed eigenvalues of the matrix Vx are of the form (normal_modes)2*m.
        m = self.m
        normal_modes = sqrt(eigenvalues/m)
        return normal_modes

def main():
    #C=(4*pi*constants.epsilon_0)**(-1)*constants.e**2
    c=classicalHamiltonian()
    #print c.potential(array([-0.5, 0.5]))
    xopt = optimize.fmin(c.potential, c.initialposition(), xtol = 1e-10)
    hessian = nd.Hessian(c.potential, stepMax=1e-1)
    H = hessian(xopt)
    hessian.error_estimate

    eigenvalues = linalg.eigvals(H)
    normal_modes = c.normal_modes(eigenvalues)

if __name__=='__main__':
    main()