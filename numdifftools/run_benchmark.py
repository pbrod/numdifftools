import numpy as np
import time

import numdifftools as nd
import numdifftools.nd_algopy as nda
import numdifftools.nd_cstep as ndc
from collections import OrderedDict
from numdifftools.nd_cstep import StepsGenerator

options = dict(step_ratio=2., step_num=15)
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

fixed_step = StepsGenerator(num_steps=1, step_ratio=4., offset=0)
epsilon = StepsGenerator(num_steps=7, step_ratio=4., offset=3)
adaptiv_txt = '_adaptive_%d_%s_%d' % (epsilon.num_steps,
                                      str(epsilon.step_ratio), epsilon.offset)
gradient_funs = OrderedDict()
gradient_funs['algopy_forward'] = lambda f: nda.Gradient(f, method='forward')
gradient_funs['numdifftools'] = lambda f: nd.Gradient(f, **options)
gradient_funs['forward'] = lambda f: ndc.Gradient(f, method='forward', steps=fixed_step)
gradient_funs['forward'+adaptiv_txt] = lambda f: ndc.Gradient(f, method='forward', steps=epsilon)
gradient_funs['central'] = lambda f: ndc.Gradient(f, method='central', steps=fixed_step)
gradient_funs['central'+adaptiv_txt] = lambda f: ndc.Gradient(f, method='central', steps=epsilon)
gradient_funs['complex'] = lambda f: ndc.Gradient(f, method='complex')
gradient_funs['complex'+adaptiv_txt] = lambda f: ndc.Gradient(f, method='complex', steps=epsilon)

hessian_funs = OrderedDict()
hessian_funs['algopy_forward'] = lambda f: nda.Hessian(f, method='forward')
hessian_funs['numdifftools'] = lambda f: nd.Hessian(f, **options)
hessian_funs['forward'] = lambda f: ndc.Hessian(f, method='forward', steps=fixed_step)
hessian_funs['forward'+adaptiv_txt] = lambda f: ndc.Hessian(f, method='forward', steps=epsilon)
hessian_funs['central'] = lambda f: ndc.Hessian(f, method='central', steps=fixed_step)
hessian_funs['central'+adaptiv_txt] = lambda f: ndc.Hessian(f, method='central', steps=epsilon)
hessian_funs['complex'] = lambda f: ndc.Hessian(f, method='complex')
hessian_funs['complex'+adaptiv_txt] = lambda f: ndc.Hessian(f, method='complex', steps=epsilon)


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
        val = gradient_f(3 * np.ones(N))
        run_time = time.time() - t
        if ref_g is None:
            ref_g = val
            err = 0
        else:
            err = np.linalg.norm(val - ref_g) / np.linalg.norm(ref_g)
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
        val = hessian_f(3 * np.ones(N))
        run_time = time.time() - t
        if ref_h is None:
            ref_h = val
            err = 0.0
        else:
            err = np.linalg.norm((val - ref_h).ravel()) / np.linalg.norm(ref_h.ravel())
        results_hessian[i] = run_time, err, preproc_time
    results_hessian_list.append(results_hessian)


results_hessians = np.array(results_hessian_list) + 1e-16

print(hessian_N_list)
print('results_hessians=\n', results_hessians)


# PLOT RESULTS

print(results_gradients.shape)

import matplotlib.pyplot as pyplot
# import prettyplotting
symbols = ['-kx','-k+', ':k>', ':k<', '--k^', '--kv', '-kp', '-ks']

plottime = pyplot.plot
ploterror = pyplot.semilogy
# plot gradient run times
for title, funcs, results in [('Gradient run times', gradient_funs, results_gradients[..., 0].T),
                       ('Hessian run times',hessian_funs, results_hessians[..., 0].T)]:
    ref_sol = results[0]
    pyplot.figure()
    pyplot.title(title)

    for i, method in enumerate(funcs):
        plottime(problem_sizes, results[i] / ref_sol,
                 symbols[i], markerfacecolor='None', label=method)
    pyplot.ylabel('Relative time $t$')
    pyplot.xlabel('problem size $N$')
    pyplot.grid()
    leg = pyplot.legend(loc=0)
    frame = leg.get_frame()
    frame.set_alpha(0.4)
    pyplot.gca().set_ylim(0, 10)
    pyplot.savefig(title.lower().replace(' ', '_') + '.png', format='png')


# # plot gradient preprocessing times
# pyplot.figure()
# pyplot.title('Gradient preprocessing times')
# plottime(problem_sizes, results_gradients[
#             :, method['numdifftools'], 2], '--ks', markerfacecolor='None',
#             label='numdifftools')
# plottime(problem_sizes, results_gradients[
#             :, method['scientific'], 2], '-.k+', markerfacecolor='None',
#             label='scientific')
# # plottime(problem_sizes, results_gradients[:,method['algopy_reverse'],2],
# #         '-.k<', markerfacecolor='None', label = 'nda reverse')
# plottime(problem_sizes, results_gradients[:, method[
#             'algopy_forward'], 2], '-.k>', markerfacecolor='None',
#             label='nda forward')
#
# pyplot.ylabel('time $t$ [seconds]')
# pyplot.xlabel('problem size $N$')
# pyplot.grid()
# leg = pyplot.legend(loc=0)
# frame = leg.get_frame()
# frame.set_alpha(0.4)
# pyplot.savefig('gradient_preprocessingtimes.png', format='png')

# plot hessian preprocessing times
# pyplot.figure()
# pyplot.title('Hessian preprocessing times')
# plottime(hessian_N_list, results_hessians[:, method['scientific'], 2],
#             '-.k+', markerfacecolor='None', label='scientific')
# plottime(hessian_N_list, results_hessians[:, method['algopy_forward'], 2],
#             '-.k>', markerfacecolor='None', label='nda (fo)')
# # plottime(hessian_N_list, results_hessians[:,method['algopy_reverse'],2],
# #  '-.k<', markerfacecolor='None', label = 'nda (fo/rev)')
# plottime(hessian_N_list, results_hessians[:, method['numdifftools'], 2],
#             '--ks', markerfacecolor='None', label='numdifftools')
# pyplot.ylabel('time $t$ [seconds]')
# pyplot.xlabel('problem size $N$')
# pyplot.grid()
# leg = pyplot.legend(loc=0)
# frame = leg.get_frame()
# frame.set_alpha(0.4)
# pyplot.savefig('hessian_preprocessingtimes.png', format='png')

# plot gradient errors
for title, funcs, results in [('Gradient errors', gradient_funs, results_gradients[..., 1].T),
                       ('Hessian errors', hessian_funs, results_hessians[..., 1].T)]:

    pyplot.figure()
    pyplot.title(title)
    ref_sol = results[0]
    for i, method in enumerate(funcs):
        ploterror(problem_sizes, results[i],
                  symbols[i], markerfacecolor='None', label=method)

    pyplot.ylabel(r'Absolute error $\|g_{ref} - g\|$')
    pyplot.xlabel('problem size $N$')
    pyplot.grid()
    leg = pyplot.legend(loc=0)
    frame = leg.get_frame()
    frame.set_alpha(0.4)
    pyplot.savefig(title.lower().replace(' ', '_') + '.png', format='png')
