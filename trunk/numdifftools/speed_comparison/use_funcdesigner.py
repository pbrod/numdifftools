import numpy as np
import FuncDesigner

class EVAL:
    def __init__(self, f, x, test = 'f'):
        self.f = f
        self.x = x.copy()

        # sA = FuncDesigner.oovar('A',shape=(len(x),len(x)))
        sx = FuncDesigner.oovar('x', size = len(x))
        sy = 0.5*FuncDesigner.dot(sx*sx,FuncDesigner.dot(f.A,sx))

        print 'sy=',sy
        
        # self.sA = sA
        self.sx = sx
        self.sy = sy
        
    def function(self, x):
        point = {self.sx:x }
        return self.sy(point)
        
    def gradient(self, x):
        point = {self.sx:x}
        # print point
        # print self.sy
        retval = self.sy.D(point)[self.sx]
        return retval
        
    # def hessian(self, x):
    #     return adolc.hessian(0,x)
     
