from __future__ import absolute_import, print_function
import timeit
from datetime import datetime
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

import numdifftools as nd
import numdifftools.nd_statsmodels as nds
import numdifftools.nd_scipy as nsc
try:
    from algopy import dot
except ImportError:
    nda = None
    dot = np.dot
else:
    import numdifftools.nd_algopy as nda
from numdifftools.core import MinStepGenerator, MaxStepGenerator


class BenchmarkFunction(object):

    """Return 0.5 * np.dot(x**2, np.dot(A,x))"""

    def __init__(self, n):
        A = np.arange(n * n, dtype=float).reshape((n, n))
        self.A = np.dot(A.T, A)

    def __call__(self, xi):
        x = np.array(xi)
        return 0.5 * dot(x * x, dot(self.A, x))


def _plot(plot, problem_sizes, objects, symbols, ylabel='', loc=2, logx=False):

    now = datetime.now().isoformat().rpartition(':')[0]

    for title, funcs, results in objects:
        plt.figure()
        plt.title(title + ' ' + now)
        for i, method in enumerate(funcs):
            plot(problem_sizes, results[i], symbols[i],
                 markerfacecolor='None', label=method)

        plt.ylabel(ylabel)
        plt.xlabel('problem size $N$')
        if logx:
            plt.xlim(loglimits(problem_sizes))
        plt.ylim(loglimits(results))
        plt.grid()
        leg = plt.legend(loc=loc)
        frame = leg.get_frame()
        frame.set_alpha(0.4)
        plt.savefig(title.lower().replace(' ', '_') + '.png', format='png')


def plot_errors(error_objects, problem_sizes, symbols):
    _plot(plt.semilogy, problem_sizes, error_objects, symbols,
          ylabel=r'Absolute error $\|g_{ref} - g\|$', loc=7, logx=False)


def plot_runtimes(run_time_objects, problem_sizes, symbols):
    _plot(plt.loglog, problem_sizes, run_time_objects, symbols,
          ylabel='time $t$', loc=2, logx=True)


def loglimits(data, border=0.05):
    low, high = np.min(data), np.max(data)
    scale = (high / low) ** border
    return low / scale, high * scale


def _compute_benchmark(functions, problem_sizes):
    result_list = []
    for n in problem_sizes:
        print('n=', n)
        num_methods = len(functions)
        results = np.zeros((num_methods, 3))
        ref_g = None
        f = BenchmarkFunction(n)
        for i, (_key, function) in enumerate(functions.items()):
            t = timeit.default_timer()
            function.fun = f
            preproc_time = timeit.default_timer() - t
            t = timeit.default_timer()
            x = 3 * np.ones(n)
            val = function(x)
            run_time = timeit.default_timer() - t
            if ref_g is None:
                ref_g = val
                err = 0
                norm_ref_g = np.linalg.norm(ref_g)
            else:
                err = np.linalg.norm(val - ref_g) / norm_ref_g
            results[i] = run_time, err, preproc_time

        result_list.append(results)

    return np.array(result_list) + 1e-16


def compute_gradients(gradient_funs, problem_sizes):
    print('starting gradient computation ')
    results_gradients = _compute_benchmark(gradient_funs, problem_sizes)
    print(list(gradient_funs))
    print('run_time, err, preproc_time')
    print('results_gradients=\n', results_gradients)
    return results_gradients


def compute_hessians(hessian_funs, problem_sizes):
    print('starting hessian computation ')
    results_hessians = _compute_benchmark(hessian_funs, problem_sizes)
    print(problem_sizes)
    print(list(hessian_funs))
    print('run_time, err, preproc_time')
    print('results_hessians=\n', results_hessians)
    return results_hessians


def run_gradient_and_hessian_benchmarks(gradient_funs, hessian_funs,
                                        problem_sizes=(4, 8, 16, 32, 64, 96),
                                        symbols=None):

    if symbols is None:
        symbols = ('-kx', ':k>', ':k<', '--k^', '--kv', '-kp', '-ks',
                   'b', '--b', '-b+', 'r', '--r', '-r+')
    results_gradients = compute_gradients(gradient_funs, problem_sizes)
    results_hessians = compute_hessians(hessian_funs, problem_sizes)

    print(results_gradients.shape)

    for i, txt in enumerate(['run times', 'errors']):
        objects = [('Jacobian ' + txt, gradient_funs, results_gradients[..., i].T),
                   ('Hessian ' + txt, hessian_funs, results_hessians[..., i].T)]
        if i == 0:
            plot_runtimes(objects, problem_sizes, symbols)
        else:
            plot_errors(objects, problem_sizes, symbols)


def main(problem_sizes=(4, 8, 16, 32, 64, 96)):
    fixed_step = MinStepGenerator(num_steps=1, use_exact_steps=True, offset=0)
    epsilon = MaxStepGenerator(num_steps=14, use_exact_steps=True,
                               step_ratio=1.6, offset=0)
    adaptiv_txt = '_adaptive_{0:d}_{1!s}_{2:d}'.format(epsilon.num_steps,
                                                       str(epsilon.step_ratio),
                                                       epsilon.offset)
    gradient_funs = OrderedDict()
    hessian_funs = OrderedDict()

    hessian_fun = 'Hessdiag'  # 'Hessian'

    if nda is not None:
        nda_method = 'forward'
        nda_txt = 'algopy_' + nda_method
        gradient_funs[nda_txt] = nda.Jacobian(1, method=nda_method)

        hessian_funs[nda_txt] = getattr(nda, hessian_fun)(1, method=nda_method)
    ndc_hessian = getattr(nd, hessian_fun)

    order = 2

    for method in ['forward', 'central', 'complex']:
        options = dict(method=method, order=order)
        gradient_funs[method] = nd.Jacobian(1, step=fixed_step, **options)
        hessian_funs[method] = ndc_hessian(1, step=fixed_step, **options)
    for method in ['forward', 'central', 'complex']:
        method2 = method + adaptiv_txt
        options = dict(method=method, order=order)
        gradient_funs[method2] = nd.Jacobian(1, step=epsilon, **options)
        hessian_funs[method2] = ndc_hessian(1, step=epsilon, **options)

    hessian_funs['forward_statsmodels'] = nds.Hessian(1, method='forward')
    hessian_funs['central_statsmodels'] = nds.Hessian(1, method='central')
    hessian_funs['complex_statsmodels'] = nds.Hessian(1, method='complex')

    gradient_funs['forward_statsmodels'] = nds.Jacobian(1, method='forward')
    gradient_funs['central_statsmodels'] = nds.Jacobian(1, method='central')
    gradient_funs['complex_statsmodels'] = nds.Jacobian(1, method='complex')

    gradient_funs['forward_scipy'] = nsc.Jacobian(1, method='forward')
    gradient_funs['central_scipy'] = nsc.Jacobian(1, method='central')
    gradient_funs['complex_scipy'] = nsc.Jacobian(1, method='complex')

    symbols = ('-kx',  # algopy
               '--m', '-m', '-m+',  # fixed: forward, central, complex
               '--g', '-g', '-g+',  # adaptive: forward, central, complex
               '--b', '-b', '-b+',  # statsmodels: forward, central, complex
               '--r', '-r', '-r+')  # scipy: forward, central, complex
    run_gradient_and_hessian_benchmarks(gradient_funs, hessian_funs, problem_sizes, symbols)


if __name__ == '__main__':
    main(problem_sizes=(4, 8, 16, 32, 64, 96))
