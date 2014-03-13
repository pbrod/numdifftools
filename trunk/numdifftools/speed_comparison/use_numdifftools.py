import numpy as np
import numdifftools as nd

class EVAL:
    def __init__(self, f, x, test = 'f'):
        self.f = f
        self.x = x
        
        self.g = nd.Gradient(f)

        self.H = nd.Hessian(f)
        
    def gradient(self, x):
        return self.g(x)

    def hessian(self, x):
        return self.H(x)











