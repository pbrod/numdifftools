import numpy as np


class F:
    def __init__(self,N):
        A = np.arange(N*N,dtype=float).reshape((N,N))
        self.A = np.dot(A.T,A)

    def __call__(self, xi):
        x = np.array(xi)
        
        tmp = np.dot(self.A,x) #np.array([(x*self.A[i]).sum() for i in range(self.A.shape[0])]) 
        return 0.5*np.dot(x*x, tmp)
        
  