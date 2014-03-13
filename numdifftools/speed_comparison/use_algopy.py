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
        return self.cg.gradient(x)

    # def hessian(self, x):
    #     return adolc.hessian(0,x)
    
    
    
class EVAL2:
    def __init__(self, f, x, test = 'f'):
        self.f = f
        self.x = x.copy()
        
        if test != 'fg' and test != 'fh':
            cg = algopy.CGraph()
            x = algopy.Function(x)
            y = f(x)
            cg.trace_off()
            cg.independentFunctionList = [x]
            cg.dependentFunctionList = [y]
            self.cg = cg
        
        
    def function(self, x):
        return self.cg.function([x])
        
    def gradient(self, x):
        return self.cg.gradient([x])
        
    def forwardgradient(self, x):
        tmp = algopy.UTPM.init_jacobian(x)
        return algopy.UTPM.extract_jacobian(self.f(tmp))

    def hessian(self, x):
        return self.cg.hessian([x])
    
    def forwardhessian(self, x):
        tmp = algopy.UTPM.init_hessian(x)
        return algopy.UTPM.extract_hessian(len(x), self.f(tmp))






 
