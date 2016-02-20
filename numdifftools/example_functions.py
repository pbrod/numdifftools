"""
Created on 17. mai 2015

@author: pab
"""
from __future__ import division
import numpy as np

function_names = ['cos', 'sin', 'tan',
                  'cosh', 'sinh', 'tanh',
                  'arcsinh',
                  'exp', 'expm1', 'exp2', 'square',
                  'sqrt',
                  'log', 'log1p', 'log10', 'log2',
                  'arccos', 'arcsin', 'arctan', ]


def dcos(x):
    return -np.sin(x)


def ddcos(x):
    return -np.cos(x)


def darcsin(x):
    return 1./np.sqrt(1-x**2)


def ddarcsin(x):
    return x/(1-x**2)**(3./2)


def dddarcsin(x):
    return 1./(1-x**2)**(3./2) + 3*x**2./(1-x**2)**(5./2)


def get_function(fun_name, n=1):

    sinh, cosh, tanh = np.sinh, np.cosh, np.tanh
    sin, cos, tan = np.sin, np.cos, np.tan
    f_dic = dict(sinh=(sinh, cosh) * 6,
                 cosh=(cosh, sinh) * 6,
                 arccosh=(np.arccosh,
                          lambda x: 1./np.sqrt(x**2-1),
                          lambda x: -x/(x**2-1)**(1.5),
                          lambda x: -1./(x**2-1)**(1.5) +
                          3*x**2/(x**2-1)**(2.5),
                          ),
                 arcsinh=(np.arcsinh,
                          lambda x: 1./np.sqrt(1+x**2),
                          lambda x: -x/(1+x**2)**(3./2),
                          lambda x: -1./(1+x**2)**(3./2) +
                          3*x**2/(1+x**2)**(5./2),
                          ),
                 arctanh=(np.arctanh,
                          lambda x: 1./(1-x**2),
                          lambda x: 2*x/(1-x**2)**2,
                          lambda x: 2./(1-x**2)**2 +
                          8*x**2/(1-x**2)**3,
                          ),
                 arccos=(np.arccos,
                         lambda x: -1./np.sqrt(1-x**2),
                         lambda x: -x/(1-x**2)**(3./2),
                         lambda x: -1./(1-x**2)**(3./2) -
                         3*x**2/(1-x**2)**(5./2),
                         ),
                 arcsin=(np.arcsin, darcsin, ddarcsin, dddarcsin),
                 square=(lambda x: x * x,  # np.square,
                         lambda x: 2 * x,
                         lambda x: 2 * np.ones_like(x)) + (
                         lambda x: np.zeros_like(x),)*15,
                 exp=(np.exp,)*20,
                 expm1=(np.expm1,) + (np.exp,)*20,
                 exp2=(np.exp2,
                       lambda x: np.exp2(x)*np.log(2),
                       lambda x: np.exp2(x)*np.log(2)**2,
                       lambda x: np.exp2(x)*np.log(2)**3,
                       lambda x: np.exp2(x)*np.log(2)**4
                       ),
                 arctan=(np.arctan,
                         lambda x: 1./(1+x**2),
                         lambda x: -2*x/(1+x**2)**2,
                         lambda x: 8.0*x**2/(1+x**2)**3 - 2./(1+x**2)**2,
                         lambda x: 24*x/(1+x**2)**3 - 48*x**3./(1+x**2)**4,
                         ),
                 cos=(cos, dcos, ddcos, sin) * 6,
                 sin=(sin, np.cos, dcos, ddcos) * 6,
                 tan=(tan,
                      lambda x: 1./np.cos(x)**2,
                      lambda x: 2*np.tan(x)/np.cos(x)**2,
                      lambda x: (4*(tan(x)**2 + 1)*tan(x)**2 +
                                 2*(tan(x)**2 + 1)**2),
                      lambda x: (8*(tan(x)**2 + 1)*tan(x)**3 +
                                 16*(tan(x)**2 + 1)**2*tan(x))
                      ),
                 tanh=(tanh,
                       lambda x: 1. / cosh(x) ** 2,
                       lambda x: -2 * sinh(x) / cosh(x) ** 3,
                       lambda x: 4*(tanh(x)/cosh(x))**2 - 2./cosh(x)**4,
                       lambda x: (8*(tanh(x)**2 - 1)*tanh(x)**3 +
                                  16*(tanh(x)**2 - 1)**2*tanh(x))),
                 log1p=(np.log1p,
                        lambda x: 1. / (1+x),
                        lambda x: -1. / (1+x) ** 2,
                        lambda x: 2. / (1+x) ** 3,
                        lambda x: -6. / (1+x) ** 4),
                 log2=(np.log2,
                       lambda x: 1. / (x*np.log(2)),
                       lambda x: -1. / (x ** 2 * np.log(2)),
                       lambda x: 2. / (x ** 3 * np.log(2)),
                       lambda x: -6. / (x ** 4 * np.log(2))),
                 log10=(np.log10,
                        lambda x: 1. / (x * np.log(10)),
                        lambda x: -1. / (x ** 2 * np.log(10)),
                        lambda x: 2. / (x ** 3 * np.log(10)),
                        lambda x: -6. / (x ** 4 * np.log(10))),
                 log=(np.log,
                      lambda x: 1. / x,
                      lambda x: -1. / x ** 2,
                      lambda x: 2. / x ** 3,
                      lambda x: -6. / x ** 4),
                 sqrt=(np.sqrt,
                       lambda x: 0.5/np.sqrt(x),
                       lambda x: -0.25/x**(1.5),
                       lambda x: 1.5*0.25/x**(2.5),
                       lambda x: -2.5*1.5*0.25/x**(3.5)),
                 inv=(lambda x: 1. / x,
                      lambda x: -1. / x ** 2,
                      lambda x: 2. / x ** 3,
                      lambda x: -6. / x ** 4,
                      lambda x: 24. / x ** 5))
    if fun_name == 'all':
        return f_dic.keys()

    funs = f_dic.get(fun_name)
    fun0 = funs[0]
    if n < len(funs):
        return fun0, funs[n]

    return fun0, None

if __name__ == '__main__':
    pass
