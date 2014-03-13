import numpy as np
import time

import benchmark1 #@UnresolvedImport
import numdifftools #@UnresolvedImport
import numdifftools.nd_algopy as algopy #@UnresolvedImport
import numdifftools.nd_scientific as scientific #@UnresolvedImport
 

method = {'numdifftools':0, 'scientific':1, 'algopy_reverse':2, 'algopy_forward':3}


 
 
# GRADIENT COMPUTATION
# --------------------

gradient_N_list = [2,4,8,16,32,64,96]
# gradient_N_list = [20]

results_gradient_list = []
for N in gradient_N_list:
    print 'N=',N
    results_gradient = np.zeros((4,3))
    # algopy, UTPS variant
    f = benchmark1.F(N)
    f0 = f(3*np.ones(N))
    t = time.time();  gradient = algopy.Gradient(f, method='forward'); preproc_time = time.time() - t
    t = time.time();  ref_g =  gradient(3*np.ones(N));  run_time = time.time() - t
    results_gradient[method['algopy_reverse']] = run_time,  0.0, preproc_time
        

    # scientifc
    f = benchmark1.F(N)
    t = time.time();  gradient = scientific.Gradient(f); preproc_time = time.time() - t
    t = time.time();  g =  gradient(3*np.ones(N));  run_time = time.time() - t
    results_gradient[method['scientific']] = run_time,  np.linalg.norm(g - ref_g)/np.linalg.norm(ref_g), preproc_time
    
   
    # algopy, UTPM variant
    f = benchmark1.F(N)
    t = time.time();  gradient = algopy.Gradient(f, method='forward'); preproc_time = time.time() - t
    t = time.time();  g =  gradient(3*np.ones(N));  run_time = time.time() - t
    results_gradient[method['algopy_forward']] = run_time,  np.linalg.norm(g - ref_g)/np.linalg.norm(ref_g), preproc_time
   
    
    # numdifftools
    f = benchmark1.F(N)
    t = time.time();  gradient = numdifftools.Gradient(f); preproc_time = time.time() - t
    t = time.time();  g =  gradient(3*np.ones(N));  run_time = time.time() - t
    results_gradient[method['numdifftools']] = run_time,  np.linalg.norm(g - ref_g)/np.linalg.norm(ref_g), preproc_time
      
     
    
    results_gradient_list.append(results_gradient)

results_gradients = np.array(results_gradient_list)+1e-18
print 'results_gradients=\n',results_gradients

# HESSIAN COMPUTATION
# -------------------
print 'starting hessian computation '
results_hessian_list = []
hessian_N_list = [1,2,4,8,16,32,64]
# hessian_N_list = [2]

for N in hessian_N_list:
    print 'N=',N
    results_hessian = np.zeros((4,3))
    
    f = benchmark1.F(N)
    t = time.time();  hessian = algopy.Hessian(f,  method='forward'); preproc_time = time.time() - t
    t = time.time();  ref_H = hessian(3*np.ones(N));  run_time = time.time() - t
    results_hessian[method['algopy_reverse']] = run_time, 0.0, preproc_time    
    
#    
    # Scientific
    f = benchmark1.F(N)
    t = time.time();  hessian = scientific.Hessian(f); preproc_time = time.time() - t
    t = time.time();  H =  hessian(3*np.ones(N));  run_time = time.time() - t
    results_hessian[method['scientific']] = run_time, np.linalg.norm( (H-ref_H).ravel())/ np.linalg.norm( (ref_H).ravel()), preproc_time
     
   
    # algopy forward utpm variant
    f = benchmark1.F(N)
    t = time.time();  hessian = algopy.Hessian(f, method='forward'); preproc_time = time.time() - t
    t = time.time();  H = hessian(3*np.ones(N));  run_time = time.time() - t
    results_hessian[method['algopy_forward']] = run_time, np.linalg.norm( (H-ref_H).ravel())/ np.linalg.norm( (ref_H).ravel()), preproc_time
    
    # numdifftools
    f = benchmark1.F(N)
    t = time.time();  hessian = numdifftools.Hessian(f); preproc_time = time.time() - t
    t = time.time();  H =  hessian(3*np.ones(N));  run_time = time.time() - t
    results_hessian[method['numdifftools']] = run_time, np.linalg.norm( (H-ref_H).ravel())/ np.linalg.norm( (ref_H).ravel()), preproc_time
    
    results_hessian_list.append(results_hessian)
    

results_hessians = np.array(results_hessian_list)+1e-18

print hessian_N_list
print 'results_hessians=\n',results_hessians


# PLOT RESULTS

print results_gradients.shape

import matplotlib.pyplot as pyplot
#import prettyplotting

plottimefun = pyplot.plot
plotfun = pyplot.semilogy
# plot gradient run times
pyplot.figure()
pyplot.title('Gradient run times')
plottimefun(gradient_N_list, results_gradients[:,method['scientific'],0], '-.k+', markerfacecolor='None', label = 'scientific')
#plottimefun(gradient_N_list, results_gradients[:,method['algopy_reverse'],0], '-.k<', markerfacecolor='None', label = 'algopy reverse utps')
plottimefun(gradient_N_list, results_gradients[:,method['algopy_forward'],0], '-.k>', markerfacecolor='None', label = 'algopy forward utpm')
plottimefun(gradient_N_list, results_gradients[:,method['numdifftools'],0], '--ks', markerfacecolor='None', label = 'numdifftools')
pyplot.ylabel('time $t$ [seconds]')
pyplot.xlabel('problem size $N$')
pyplot.grid()
leg = pyplot.legend(loc=0)
frame= leg.get_frame()
frame.set_alpha(0.4)
pyplot.savefig('gradient_runtimes.png',format='png')

# plot hessian run times
pyplot.figure()
pyplot.title('Hessian run times')
plottimefun(hessian_N_list, results_hessians[:,method['scientific'],0], '-.k+', markerfacecolor='None', label = 'scientific')
plottimefun(hessian_N_list, results_hessians[:,method['algopy_forward'],0], '-.k>', markerfacecolor='None', label = 'algopy (fo)')
#plottimefun(hessian_N_list, results_hessians[:,method['algopy_reverse'],0], '-.k<', markerfacecolor='None', label = 'algopy (fo/rev)')
plottimefun(hessian_N_list, results_hessians[:,method['numdifftools'],0], '--ks', markerfacecolor='None', label = 'numdifftools')
pyplot.ylabel('time $t$ [seconds]')
pyplot.xlabel('problem size $N$')
pyplot.grid()
leg = pyplot.legend(loc=0)
frame= leg.get_frame()
frame.set_alpha(0.4)
pyplot.savefig('hessian_runtimes.png',format='png')


# plot gradient preprocessing times
pyplot.figure()
pyplot.title('Gradient preprocessing times')
plottimefun(gradient_N_list, results_gradients[:,method['numdifftools'],2], '--ks', markerfacecolor='None', label = 'numdifftools')
plottimefun(gradient_N_list, results_gradients[:,method['scientific'],2], '-.k+', markerfacecolor='None', label = 'scientific')
#plottimefun(gradient_N_list, results_gradients[:,method['algopy_reverse'],2], '-.k<', markerfacecolor='None', label = 'algopy reverse')
plottimefun(gradient_N_list, results_gradients[:,method['algopy_forward'],2], '-.k>', markerfacecolor='None', label = 'algopy forward')

pyplot.ylabel('time $t$ [seconds]')
pyplot.xlabel('problem size $N$')
pyplot.grid()
leg = pyplot.legend(loc=0)
frame= leg.get_frame()
frame.set_alpha(0.4)
pyplot.savefig('gradient_preprocessingtimes.png',format='png')

# plot hessian preprocessing times
pyplot.figure()
pyplot.title('Hessian preprocessing times')
plottimefun(hessian_N_list, results_hessians[:,method['scientific'],2], '-.k+', markerfacecolor='None', label = 'scientific')
plottimefun(hessian_N_list, results_hessians[:,method['algopy_forward'],2], '-.k>', markerfacecolor='None', label = 'algopy (fo)')
#plottimefun(hessian_N_list, results_hessians[:,method['algopy_reverse'],2], '-.k<', markerfacecolor='None', label = 'algopy (fo/rev)')
plottimefun(hessian_N_list, results_hessians[:,method['numdifftools'],2], '--ks', markerfacecolor='None', label = 'numdifftools')
pyplot.ylabel('time $t$ [seconds]')
pyplot.xlabel('problem size $N$')
pyplot.grid()
leg = pyplot.legend(loc=0)
frame= leg.get_frame()
frame.set_alpha(0.4)
pyplot.savefig('hessian_preprocessingtimes.png',format='png')

# plot gradient errors
pyplot.figure()
pyplot.title('Gradient Correctness')
plotfun(gradient_N_list, results_gradients[:,method['numdifftools'],1], '--ks', markerfacecolor='None', label = 'numdifftools')
plotfun(gradient_N_list, results_gradients[:,method['scientific'],1], '-.k+', markerfacecolor='None', label = 'scientific')
#plotfun(gradient_N_list, results_gradients[:,method['algopy_reverse'],1], '-.k<', markerfacecolor='None', label = 'algopy reverse')
plotfun(gradient_N_list, results_gradients[:,method['algopy_forward'],1], '-.k>', markerfacecolor='None', label = 'algopy forward')
pyplot.ylabel(r'relative error $\|g_{ref} - g\|/\|g_{ref}\}$')
pyplot.xlabel('problem size $N$')
pyplot.grid()
leg = pyplot.legend(loc=0)
frame= leg.get_frame()
frame.set_alpha(0.4)
pyplot.savefig('gradient_errors.png',format='png')

# plot hessian errors
pyplot.figure()
pyplot.title('Hessian Correctness')
plotfun(hessian_N_list, results_hessians[:,method['numdifftools'],1], '--ks', markerfacecolor='None', label = 'numdifftools')
plotfun(hessian_N_list, results_hessians[:,method['scientific'],1], '-.k+', markerfacecolor='None', label = 'scientific')
plotfun(hessian_N_list, results_hessians[:,method['algopy_forward'],1], '-.k>', markerfacecolor='None', label = 'algopy (fo)')
#plotfun(hessian_N_list, results_hessians[:,method['algopy_reverse'],1], '-.k<', markerfacecolor='None', label = 'algopy (fo/rev)')
pyplot.ylabel(r'relative error $\|H_{ref} - H\|/\|H_{ref}\|$')
pyplot.xlabel(r'problem size $N$')
pyplot.grid()
leg = pyplot.legend(loc=0)
frame= leg.get_frame()
frame.set_alpha(0.4)
pyplot.savefig('hessian_errors.png',format='png')



# pyplot.show()


