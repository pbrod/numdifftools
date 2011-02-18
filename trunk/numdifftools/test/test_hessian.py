# import a tool to use / as a symbol for normal division
from __future__ import division

#import system data
import sys, os

#Loading the required packages
import scipy as sp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# and subpackages
from scipy import *
from scipy import linalg, optimize, constants

import numpy
import numdifftools as nd
import time

from time import gmtime, strftime
def now():
    '''
    Return current date and time as a string
    '''
    return strftime("%a, %d %b %Y %H:%M:%S", gmtime())
#-----------------------------------------------------------------------------------------
#       Hamiltonian      H=sum_i(p_i^2/(2m)+ 1/2 * m * w^2 x_i^2)+ sum_(i!=j)(a/|x_i-x_j|)
#-----------------------------------------------------------------------------------------

class classicalHamiltonian:
   def __init__(self):

       self.N = 2                                                      #N is a scalar, it's the number of ions in the chain
       f = 1000000                                                     #f is a scalar, it's the trap frequency
       self.w = 2*pi*f                                                 #w is a scalar, it's the angular velocity corresponding to the trap frequency
       self.C = (4*pi*constants.epsilon_0)**(-1)*constants.e**2        #C is a scalar, it's the Coulomb constant times the electronic charge in SI
       self.m = 39.96*1.66e-27                                         #m is the mass of a single trapped ion in the chain

   def potential(self, positionvector):                                    #Defines the potential that is going to be minimized

       x= positionvector                                               #x is an 1-d array (vector) of lenght N that contains the positions of the N ions
       w= self.w
       C= self.C
       m= self.m

       #First we consider the potential of the harmonic osszilator
       Vx = 0.5 * m * (w**2) * sum(x**2)

       for i in range(len(x)):
           for j in range(i+1, len(x)):
               Vx += C * (abs(x[i]-x[j]))**(-1)        #then we add the coulomb interaction

       return Vx

   def initialposition(self):              #Defines the initial position as an estimate for the minimize process

       N= self.N
       x_0 = r_[-(N-1)/2:(N-1)/2:N*1j]
       return x_0

   def normal_modes(self, eigenvalues):    #the computed eigenvalues of the matrix Vx are of the form (normal_modes)^2*m.
       m = self.m
       normal_modes = sqrt(eigenvalues/m)
       return normal_modes

c=classicalHamiltonian()
xopt = optimize.fmin(c.potential, c.initialposition(), xtol = 1e-10)

t0 = time.clock()
hessian = nd.Hessian(c.potential, stepMax=1e-3)
H = hessian(xopt)
print(time.clock()-t0)

print(H)
print('Error:')
print(hessian.error_estimate)

#x = algopy.UTPM.init_hessian(xopt)
#y = c.potential(x)
#hessian = algopy.UTPM.extract_hessian(2, y)
#
#print hessian