import numpy as np
import uncertainties
import uncertainties.unumpy as unp


class EVAL:
    def __init__(self, f, x, test = 'f'):
        self.f = f
        
    def gradient(self, x):
        sx = unp.uarray((x, np.inf))
        sf = self.f(sx)
        return np.array([sf.derivatives[var] for var in sx])










