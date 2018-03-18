from __future__ import division, print_function
from numdifftools.example_functions import get_function, function_names
from numdifftools import Derivative
from numdifftools.step_generators import default_scale, MinStepGenerator
import numpy as np
import matplotlib.pyplot as plt


def _example3(x=0.0001, fun_name='cos', epsilon=None, method='central',
              scale=None, n=1, order=2):
    fun0, dfun = get_function(fun_name, n)
    if dfun is None:
        return dict(n=n, order=order, method=method, fun=fun_name,
                    error=np.nan, scale=np.nan)
    fd = Derivative(fun0, step=epsilon, method=method, n=n, order=order)
    t = []
    scales = np.arange(1.0, 35, 0.25)
    for scale in scales:
        fd.step.scale = scale
        try:
            val = fd(x)
        except Exception:
            val = np.nan
        t.append(val)
    t = np.array(t)
    tt = dfun(x)
    relativ_error = np.abs(t - tt) / (np.maximum(np.abs(tt), 1)) + 1e-16

#     weights = np.ones((3,))/3
#     relativ_error = convolve1d(relativ_error, weights)  # smooth curve

    if np.isnan(relativ_error).all():
        return dict(n=n, order=order, method=method, fun=fun_name,
                    error=np.nan, scale=np.nan)
    if True:  # False:  #
        plt.semilogy(scales, relativ_error, label=fun_name)
        plt.vlines(default_scale(fd.method, n, order),
                   np.nanmin(relativ_error), 1)
        plt.xlabel('scales')
        plt.ylabel('Relative error')
        txt = ['', "1'st", "2'nd", "3'rd", "4'th", "5'th", "6'th",
               "7th"] + ["%d'th" % i for i in range(8, 25)]

        plt.title("The %s derivative using %s, order=%d" % (txt[n], method,
                                                            order))
        plt.legend(frameon=False, framealpha=0.5)
        plt.axis([min(scales), max(scales), np.nanmin(relativ_error), 1])
        # plt.figure()
        # plt.show('hold')
    i = np.nanargmin(relativ_error)
    return dict(n=n, order=order, method=method, fun=fun_name,
                error=relativ_error[i], scale=scales[i])


if __name__ == '__main__':
    method = 'forward'
    order = 4
    epsilon = MinStepGenerator(num_steps=1, scale=None)
    scales = {}
    for n in range(1, 11):
        plt.figure(n)
        for x in [0.1, 0.5, 1.0, 5]:
            for name in function_names:
                r = _example3(x=x, fun_name=name, epsilon=epsilon,
                              method=method, scale=None, n=n, order=order)
                print(r)
                scale_n = scales.setdefault(n, [])
                scale = r['scale']
                if np.isfinite(scale):
                    scale_n.append(scale)
        plt.vlines(np.mean(scale_n), 1e-12, 1, 'r', linewidth=3)

    print(scales)
    print('method={}, order={}'.format(method, order))
    for n in scales:
        print('n={}, scale={}'.format(n, np.mean(scales[n])))

    print('Default scale')
    for n in scales:
        print('n={}, scale={}'.format(n, default_scale(method, n, order)))

    plt.show('hold')

# method=complex, order=2
# n=1, scale=1.05188679245
# n=2, scale=5.55424528302
# n=3, scale=8.34669811321
# n=4, scale=9.08072916667
# n=5, scale=10.1958333333
# n=6, scale=9.13020833333
# n=7, scale=13.859375
# n=8, scale=14.125
# n=9, scale=12.8385416667
# n=10, scale=14.5416666667

# method=central, order=2, x=0.5
# n=1, scale=2.57894736842
# n=2, scale=3.81578947368
# n=3, scale=5.01315789474
# n=4, scale=5.578125
# n=5, scale=6.625
# n=6, scale=7.59375
# n=7, scale=8.65625
# n=8, scale=9.28125
# n=9, scale=9.84375

# method=central, order=2, x=5
# n=1, scale=2.86764705882
# n=2, scale=5.41176470588
# n=3, scale=6.23529411765
# n=4, scale=5.859375
# n=5, scale=8.025
# n=6, scale=6.90625
# n=7, scale=7.0625
# n=8, scale=8.21875
# n=9, scale=9.9375

# method=central, order=2  x in [0.1, 0.5, 1.0, 5, 10, 50,]:
# n=1, scale=2.77358490566
# n=2, scale=4.75471698113
# n=3, scale=5.19575471698
# n=4, scale=5.7890625
# n=5, scale=7.05
# n=6, scale=7.046875
# n=7, scale=7.89583333333
# n=8, scale=8.41145833333
# n=9, scale=9.21354166667
# n=10, scale=9.33854166667
