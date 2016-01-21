from __future__ import print_function
import numpy as np
import timeit

import numdifftools as nd
import numdifftools.nd_algopy as nda
from algopy import dot
# from numpy import dot
from collections import OrderedDict
from numdifftools.core import MinStepGenerator, MaxStepGenerator
import matplotlib.pyplot as plt


class BenchmarkFunction(object):
    def __init__(self, N):
        A = np.arange(N * N, dtype=float).reshape((N, N))
        self.A = np.dot(A.T, A)

    def __call__(self, xi):
        x = np.array(xi)
        tmp = dot(self.A, x)
        return 0.5 * dot(x * x, tmp)


def plot_errors(error_objects, problem_sizes, symbols):
    ploterror = plt.semilogy
    for title, funcs, results in error_objects:
        plt.figure()
        plt.title(title)
        # ref_sol = results[0]
        for i, method in enumerate(funcs):
            ploterror(problem_sizes, results[i], symbols[i],
                      markerfacecolor='None', label=method)

        plt.ylabel(r'Absolute error $\|g_{ref} - g\|$')
        plt.xlabel('problem size $N$')
        plt.ylim(loglimits(results))
        plt.grid()
        leg = plt.legend(loc=7)
        frame = leg.get_frame()
        frame.set_alpha(0.4)
        plt.savefig(title.lower().replace(' ', '_') + '.png', format='png')


def plot_runtimes(run_time_objects, problem_sizes, symbols):
    plottime = plt.loglog
    for title, funcs, results in run_time_objects:
        plt.figure()
        plt.title(title)
        for i, method in enumerate(funcs):
            plottime(problem_sizes, results[i], symbols[i],
                     markerfacecolor='None', label=method)

        plt.ylabel('time $t$')
        plt.xlabel('problem size $N$')
        plt.xlim(loglimits(problem_sizes))
        plt.ylim(loglimits(results))
        plt.grid()
        leg = plt.legend(loc=2)
        frame = leg.get_frame()
        frame.set_alpha(0.4)
        plt.savefig(title.lower().replace(' ', '_') + '.png', format='png')


def loglimits(data, border=0.05):
    low, high = np.min(data), np.max(data)
    scale = (high/low)**border
    return low/scale, high*scale


fixed_step = MinStepGenerator(num_steps=1, use_exact_steps=True, offset=0)
epsilon = MaxStepGenerator(num_steps=14, use_exact_steps=True,
                           step_ratio=1.6, offset=0)
adaptiv_txt = '_adaptive_{0:d}_{1!s}_{2:d}'.format(epsilon.num_steps,
                                      str(epsilon.step_ratio), epsilon.offset)
gradient_funs = OrderedDict()
nda_method = 'forward'
nda_txt = 'algopy_' + nda_method
gradient_funs[nda_txt] = nda.Jacobian(1, method=nda_method)
# gradient_funs['numdifftools'] = nd.Jacobian(1, **options)
for method in ['forward', 'central', 'complex']:
    method2 = method + adaptiv_txt
    gradient_funs[method] = nd.Jacobian(1, method=method, step=fixed_step)
    gradient_funs[method2] = nd.Jacobian(1, method=method, step=epsilon)

HessianFun = 'Hessdiag'
ndcHessian = getattr(nd, HessianFun)  # ndc.Hessian #
hessian_funs = OrderedDict()
hessian_funs[nda_txt] = getattr(nda, HessianFun)(1, method=nda_method)

for method in ['forward', 'central', 'complex']:
    method2 = method + adaptiv_txt
    hessian_funs[method] = ndcHessian(1, method=method, step=fixed_step)
    hessian_funs[method2] = ndcHessian(1, method=method, step=epsilon)


def compute_gradients(gradient_funs, problem_sizes):

    results_gradient_list = []
    for N in problem_sizes:
        print('N=', N)
        num_methods = len(gradient_funs)
        results_gradient = np.zeros((num_methods, 3))
        ref_g = None
        f = BenchmarkFunction(N)
        for i, (_key, gradient_f) in enumerate(gradient_funs.items()):
            t = timeit.default_timer()
            gradient_f.f = f
            preproc_time = timeit.default_timer() - t
            t = timeit.default_timer()
            x = 3 * np.ones(N)
            val = gradient_f(x)
            run_time = timeit.default_timer() - t
            if ref_g is None:
                ref_g = val
                err = 0
                norm_ref_g = np.linalg.norm(ref_g)
            else:
                err = np.linalg.norm(val - ref_g) / norm_ref_g
            results_gradient[i] = run_time, err, preproc_time

        results_gradient_list.append(results_gradient)

    results_gradients = np.array(results_gradient_list) + 1e-16
    print('results_gradients=\n', results_gradients)
    return results_gradients


def compute_hessians(hessian_funs, problem_sizes):
    print('starting hessian computation ')
    results_hessian_list = []
    hessian_N_list = problem_sizes
    for N in hessian_N_list:
        print('N=', N)
        num_methods = len(hessian_funs)
        results_hessian = np.zeros((num_methods, 3))
        ref_h = None
        f = BenchmarkFunction(N)
        for i, (_key, hessian_f) in enumerate(hessian_funs.items()):
            t = timeit.default_timer()
            hessian_f.f = f
            preproc_time = timeit.default_timer() - t
            t = timeit.default_timer()
            x = 3 * np.ones(N)
            val = hessian_f(x)
            run_time = timeit.default_timer() - t
            if ref_h is None:
                ref_h = val
                err = 0.0
                norm_ref_h = np.linalg.norm(ref_h.ravel())
            else:
                err = np.linalg.norm((val - ref_h).ravel()) / norm_ref_h
            results_hessian[i] = run_time, err, preproc_time

        results_hessian_list.append(results_hessian)

    results_hessians = np.array(results_hessian_list) + 1e-16
    print(hessian_N_list)
    print('results_hessians=\n', results_hessians)
    return results_hessians


if __name__ == '__main__':
    problem_sizes = (4, 8, 16, 32, 64, 96)
    symbols = ('-kx', ':k>', ':k<', '--k^', '--kv', '-kp', '-ks',
               'b', '--b', '-k+')

    results_gradients = compute_gradients(gradient_funs, problem_sizes)
    results_hessians = compute_hessians(hessian_funs, problem_sizes)

    print(results_gradients.shape)

    run_time_objects = [('Jacobian run times',
                         gradient_funs, results_gradients[..., 0].T),
                        ('Hessian run times',
                         hessian_funs, results_hessians[..., 0].T)]
    error_objects = [('Jacobian errors',
                      gradient_funs, results_gradients[..., 1].T),
                     ('Hessian errors',
                      hessian_funs, results_hessians[..., 1].T)]
    plot_runtimes(run_time_objects, problem_sizes, symbols)
    plot_errors(error_objects, problem_sizes, symbols)
