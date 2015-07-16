import numpy as np
import time

import numdifftools as nd
import numdifftools.nd_algopy as nda
import numdifftools.nd_cstep as ndc
from collections import OrderedDict
from numdifftools.nd_cstep import StepGenerator, MaxStepGenerator
import matplotlib.pyplot as pyplot
# import prettyplotting

options = dict(step_ratio=2., step_num=15, vectorized=True)
method = {'numdifftools': 0, 'scientific': 1,
          'algopy_reverse': 2, 'algopy_forward': 3}


class BenchmarkFunction(object):
    def __init__(self, N):
        A = np.arange(N * N, dtype=float).reshape((N, N))
        self.A = np.dot(A.T, A)

    def __call__(self, xi):
        x = np.array(xi)
        tmp = np.dot(self.A, x)
        return 0.5 * np.dot(x * x, tmp)

fixed_step = StepGenerator(num_steps=1, use_exact_steps=True, offset=4) #, step_ratio=2., offset=3)
epsilon = StepGenerator(num_steps=7, use_exact_steps=True, offset=5) #, step_ratio=4., offset=3)
fixed_step = StepGenerator(num_steps=2, use_exact_steps=True, step_ratio=2., offset=0)
fixed_step2 = StepGenerator(num_steps=1, use_exact_steps=True, step_ratio=2, offset=0)
epsilon = StepGenerator(num_steps=10, use_exact_steps=True, step_ratio=2, offset=0)
epsilon = MaxStepGenerator(num_steps=15, use_exact_steps=False, step_ratio=2)
adaptiv_txt = '_adaptive_%d_%s_%d' % (epsilon.num_steps,
                                      str(epsilon.step_ratio), epsilon.offset)
gradient_funs = OrderedDict()
gradient_funs['algopy_forward'] = lambda f: nda.Gradient(f, method='forward')
gradient_funs['numdifftools'] = lambda f: nd.Gradient(f, **options)
gradient_funs['forward'] = lambda f: ndc.Gradient(f, method='forward',
                                                  step=fixed_step)
gradient_funs['forward'+adaptiv_txt] = lambda f: ndc.Gradient(f,
                                                              method='forward',
                                                              step=epsilon)
gradient_funs['central'] = lambda f: ndc.Gradient(f, method='central',
                                                  step=fixed_step2)
gradient_funs['central'+adaptiv_txt] = lambda f: ndc.Gradient(f,
                                                              method='central',
                                                              step=epsilon)
gradient_funs['complex'] = lambda f: ndc.Gradient(f, method='complex',
                                                  step=fixed_step2)
gradient_funs['complex'+adaptiv_txt] = lambda f: ndc.Gradient(f,
                                                              method='complex',
                                                              step=epsilon)

hessian_funs = OrderedDict()
hessian_funs['algopy_forward'] = lambda f: nda.Hessian(f, method='forward')
hessian_funs['numdifftools'] = lambda f: nd.Hessian(f, **options)
hessian_funs['forward'] = lambda f: ndc.Hessian(f, method='forward',
                                                step=fixed_step2)
hessian_funs['forward'+adaptiv_txt] = lambda f: ndc.Hessian(f,
                                                            method='forward',
                                                            step=epsilon)
hessian_funs['central2'] = lambda f: ndc.Hessian(f, method='central2',
                                                 step=fixed_step2)
hessian_funs['central2'+adaptiv_txt] = lambda f: ndc.Hessian(f,
                                                             method='central2',
                                                             step=epsilon)
hessian_funs['hybrid'] = lambda f: ndc.Hessian(f, method='hybrid',
                                               step=fixed_step2)
hessian_funs['hybrid'+adaptiv_txt] = lambda f: ndc.Hessian(f, method='hybrid',
                                                           step=epsilon)
# hessian_funs['complex'] = lambda f: ndc.Hessian(f, method='complex',
#                                                step=fixed_step)


# GRADIENT COMPUTATION
# --------------------
problem_sizes = [4, 8, 16, 32, 64, 96]

results_gradient_list = []
for N in problem_sizes:
    print('N=', N)
    num_methods = len(gradient_funs)
    results_gradient = np.zeros((num_methods, 3))
    ref_g = None
    f = BenchmarkFunction(N)
    for i, (key, Gradient) in enumerate(gradient_funs.iteritems()):
        t = time.time()
        gradient_f = Gradient(f)
        preproc_time = time.time() - t
        t = time.time()
        x = 3 * np.ones(N)
        val = gradient_f(x)
        run_time = time.time() - t
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

# HESSIAN COMPUTATION
# -------------------
print('starting hessian computation ')
results_hessian_list = []
hessian_N_list = problem_sizes

for N in hessian_N_list:
    print('N=', N)
    num_methods = len(hessian_funs)
    results_hessian = np.zeros((num_methods, 3))
    ref_h = None
    f = BenchmarkFunction(N)
    for i, (key, Hessian) in enumerate(hessian_funs.iteritems()):
        t = time.time()
        hessian_f = Hessian(f)
        preproc_time = time.time() - t
        t = time.time()
        x = 3 * np.ones(N)
        val = hessian_f(x)
        run_time = time.time() - t
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


# -- PLOT RESULTS

print(results_gradients.shape)


symbols = ['-kx', '-k+', ':k>', ':k<', '--k^', '--kv', '-kp', '-ks', 'b']

plottime = pyplot.plot
ploterror = pyplot.semilogy

#  plot gradient run times
for title, funcs, results in [('Gradient run times',
                               gradient_funs, results_gradients[..., 0].T),
                              ('Hessian run times',
                               hessian_funs, results_hessians[..., 0].T)]:
    ref_sol = results[0]
    pyplot.figure()
    pyplot.title(title)

    for i, method in enumerate(funcs):
        plottime(problem_sizes, results[i] / ref_sol,
                 symbols[i], markerfacecolor='None', label=method)
    pyplot.ylabel('Relative time $t$')
    pyplot.xlabel('problem size $N$')
    pyplot.grid()
    leg = pyplot.legend(loc=1)
    frame = leg.get_frame()
    frame.set_alpha(0.4)
    pyplot.gca().set_ylim(0, 10)
    pyplot.savefig(title.lower().replace(' ', '_') + '.png', format='png')


#  plot gradient errors
for title, funcs, results in [('Gradient errors',
                               gradient_funs, results_gradients[..., 1].T),
                              ('Hessian errors',
                               hessian_funs, results_hessians[..., 1].T)]:

    pyplot.figure()
    pyplot.title(title)
    ref_sol = results[0]
    for i, method in enumerate(funcs):
        ploterror(problem_sizes, results[i],
                  symbols[i], markerfacecolor='None', label=method)

    pyplot.ylabel(r'Absolute error $\|g_{ref} - g\|$')
    pyplot.xlabel('problem size $N$')
    pyplot.grid()
    leg = pyplot.legend(loc=1)
    frame = leg.get_frame()
    frame.set_alpha(0.4)
    pyplot.gca().set_ylim(1e-16, 1e-3)
    pyplot.savefig(title.lower().replace(' ', '_') + '.png', format='png')
