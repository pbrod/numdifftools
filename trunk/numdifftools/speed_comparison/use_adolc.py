import numpy
try:
    import adolc
except:
    adolc = None
import numpy as np
import algopy

class EVAL:
    def __init__(self, f, x, test = 'f'):
        self.f = f
        self.x = x.copy()

        cg = algopy.CGraph()
        x = np.array([algopy.Function(x[i]) for i in range(len(x))])
        y = f(x)
        # print 'y=',y
        cg.trace_off()
        cg.independentFunctionList = x
        cg.dependentFunctionList = [y]
        self.cg = cg
        
    def function(self, x):
        return self.cg.function(x)
        
    def gradient(self, x):
        return np.asarray(self.cg.gradient(x))
#    def hessian(self, x):
#        return np.asarray(self.cg.hessian([x]))
    def hessian(self, x):
        tmp = algopy.UTPM.init_hessian(x)
        return algopy.UTPM.extract_hessian(len(x), self.f(tmp))

class EVAL0:
    def __init__(self, f, x, test = 'f'):
        self.f = f
        self.x = x

        adolc.trace_on(0)
        ax = adolc.adouble(x)
        adolc.independent(ax)
        y = f(ax)
        adolc.dependent(y)
        adolc.trace_off()
        
    def function(self, x):
        return adolc.function(0,x)
        
    def gradient(self, x):
        return adolc.gradient(0,x)

    def hessian(self, x):
        return adolc.hessian(0,x)







