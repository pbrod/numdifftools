import numpy as np
import  Scientific.Functions.Derivatives as SFD


class EVAL:
    def __init__(self, f, x, test = 'f'):
        self.f = f
        self.x = x
        
    def gradient(self, x):
        N = len(x)
        sx = np.array([SFD.DerivVar(x[i], i, 1) for i in range(N)])
        return np.array(self.f(sx)[1])

    def hessian(self, x):
        N = len(x)
        sx = np.array([SFD.DerivVar(x[i], i, 2) for i in range(N)])
        return np.array(self.f(sx)[2])






