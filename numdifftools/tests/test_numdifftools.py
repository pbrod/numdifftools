""" Test functions for numdifftools module

"""
from __future__ import print_function
import unittest
import numdifftools.core as nd
import numpy as np
from numpy.testing import assert_array_almost_equal


class TestGlobalFunctions(unittest.TestCase):

    def testdea3(self):
        def linfun(k):
            return np.linspace(0, np.pi / 2., 2. ** (k + 5) + 1)
        Ei = np.zeros(3)
        for k in np.arange(3):
            x = linfun(k)
            Ei[k] = np.trapz(np.sin(x), x)
        [En, err] = nd.dea3(Ei[0], Ei[1], Ei[2])
        self.assertTrue(np.abs(En - 1) < err)
        assert_array_almost_equal(En, 1.0, decimal=8)


class TestRichardson(unittest.TestCase):

    def test_order_step_combinations(self):
        true_vals = {
            (1, 1, 1): [-0.9999999999999998, 1.9999999999999998],
            (1, 1, 2): [-0.33333333333333304, 1.333333333333333],
            (1, 1, 3): [-0.14285714285714307, 1.142857142857143],
            (1, 1, 4): [-0.06666666666666654, 1.0666666666666664],
            (1, 1, 5): [-0.03225806451612906, 1.0322580645161292],
            (1, 1, 6): [-0.015873015873015872, 1.0158730158730154],
            (1, 2, 1): [-0.9999999999999998, 1.9999999999999998],
            (1, 2, 2): [-0.33333333333333304, 1.333333333333333],
            (1, 2, 3): [-0.14285714285714307, 1.142857142857143],
            (1, 2, 4): [-0.06666666666666654, 1.0666666666666664],
            (1, 2, 5): [-0.03225806451612906, 1.0322580645161292],
            (1, 2, 6): [-0.015873015873015872, 1.0158730158730154],
            (2, 1, 1): [0.33333333333333337, -2.0, 2.666666666666667],
            (2, 1, 2): [0.04761904761904753, -0.5714285714285693,
                        1.523809523809522],
            (2, 1, 3): [0.009523809523810024, -0.2285714285714322,
                        1.2190476190476225],
            (2, 1, 4): [0.002150537634408055, -0.10322580645160284,
                        1.1010752688171945],
            (2, 1, 5): [0.0005120327700975248, -0.04915514592935677,
                        1.0486431131592595],
            (2, 1, 6): [0.0001249843769525012, -0.02399700037493191,
                        1.0238720159979793],
            (2, 2, 1): [0.1428571428571428, -1.428571428571427,
                        2.2857142857142843],
            (2, 2, 2): [0.022222222222222185, -0.444444444444444,
                        1.4222222222222216],
            (2, 2, 3): [0.004608294930875861, -0.1843317972350207,
                        1.179723502304145],
            (2, 2, 4): [0.0010582010582006751, -0.08465608465608221,
                        1.0835978835978812],
            (2, 2, 5): [0.0002540005080009476, -0.040640081280166496,
                        1.0403860807721657],
            (2, 2, 6): [6.224712107032182e-05, -0.01991907874258203,
                        1.0198568316215115],
            (3, 1, 1): [-0.04761904761904734, 0.6666666666666641,
                        -2.6666666666666594, 3.047619047619042],
            (3, 1, 2): [-0.003174603174603108, 0.08888888888889318,
                        -0.7111111111111337, 1.6253968253968432],
            (3, 1, 3): [-0.0003072196620577672, 0.01720430107525861,
                        -0.27526881720422713, 1.258371735791026],
            (3, 1, 4): [-3.4135518007183396e-05, 0.003823178016754525,
                        -0.12234169653539884, 1.1185526540366513],
            (3, 1, 5): [-4.031754094968587e-06, 0.0009031129172963892,
                        -0.0577992267083981, 1.056900145545197],
            (3, 1, 6): [-4.901348115149418e-07, 0.00021958039560535103,
                        -0.02810629063481751, 1.0278872003740238],
            (3, 2, 1): [-0.004608294930874168, 0.1935483870967698,
                        -1.5483870967741966, 2.359447004608302],
            (3, 2, 2): [-0.00035273368606647537, 0.02962962962962734,
                        -0.47407407407406155, 1.444797178130501],
            (3, 2, 3): [-3.628578685754835e-05, 0.006096012192020994,
                        -0.19507239014474764, 1.1890126637395837],
            (3, 2, 4): [-4.149808071229888e-06, 0.0013943355119737377,
                        -0.08923747276669802, 1.0878472870627958],
            (3, 2, 5): [-4.970655732572382e-07, 0.0003340280653114369,
                        -0.042755592360228634, 1.0424220613604906],
            (3, 2, 6): [-6.08476257157875e-08, 8.177920896951241e-05,
                        -0.02093547748207586, 1.0208537591207332]}
        for num_terms in [1, 2, 3]:
            for step in [1, 2]:
                for order in range(1, 7):
                    r_extrap = nd.Richardson(step_ratio=2.0, step=step,
                                             num_terms=num_terms, order=order)
                    rule = r_extrap._get_richardson_rule()
                    # print((num_terms, step, order), rule.tolist())
                    assert_array_almost_equal(rule,
                                              true_vals[(num_terms, step,
                                                         order)])
        # self.assert_(False)

    def test_central(self):
        method = 'central'
        true_vals = {(1, 1): [-0.33333333, 1.33333333],
                     (1, 2): [-0.33333333, 1.33333333],
                     (1, 3): [-0.33333333, 1.33333333],
                     (1, 4): [-0.06666667, 1.06666667],
                     (1, 5): [-0.06666667, 1.06666667],
                     (1, 6): [-0.01587302, 1.01587302],
                     (2, 1): [0.02222222, -0.44444444, 1.42222222],
                     (2, 2): [0.02222222, -0.44444444, 1.42222222],
                     (2, 3): [0.02222222, -0.44444444, 1.42222222],
                     (2, 4): [1.05820106e-03, -8.46560847e-02, 1.08359788e+00],
                     (2, 5): [1.05820106e-03, -8.46560847e-02, 1.08359788e+00],
                     (2, 6): [6.22471211e-05, -1.99190787e-02, 1.01985683e+00]}

        for num_terms in [1, 2]:
            for order in range(1, 7):
                d = nd.Derivative(np.exp, method=method, order=order)
                d._set_richardson_rule(step_ratio=2.0, num_terms=num_terms)
                rule = d._richardson_extrapolate._get_richardson_rule()
                assert_array_almost_equal(rule,
                                          true_vals[(num_terms, order)])

    def test_complex(self):
        truth = {
            (1, 2, 8): [9.576480164718605e-07, -0.004167684167715291,
                        1.004166726519699],
            (4, 2, 2): [0.0002614379084968331, -0.07111111111111235,
                        1.070849673202616],
            (1, 2, 4): [0.0002614379084968331, -0.07111111111111235,
                        1.070849673202616],
            (4, 1, 8): [-0.0039215686274510775, 1.0039215686274505],
            (2, 2, 4): [0.0002614379084968331, -0.07111111111111235,
                        1.070849673202616],
            (4, 2, 8): [9.576480164718605e-07, -0.004167684167715291,
                        1.004166726519699],
            (3, 1, 8): [-0.0039215686274510775, 1.0039215686274505],
            (4, 1, 2): [-0.06666666666666654, 1.0666666666666664],
            (3, 1, 6): [-0.06666666666666654, 1.0666666666666664],
            (1, 1, 8): [-0.0039215686274510775, 1.0039215686274505],
            (2, 1, 8): [-0.0039215686274510775, 1.0039215686274505],
            (4, 1, 4): [-0.06666666666666654, 1.0666666666666664],
            (3, 1, 4): [-0.06666666666666654, 1.0666666666666664],
            (2, 1, 4): [-0.06666666666666654, 1.0666666666666664],
            (3, 2, 2): [0.0002614379084968331, -0.07111111111111235,
                        1.070849673202616],
            (2, 2, 8): [9.576480164718605e-07, -0.004167684167715291,
                        1.004166726519699],
            (2, 1, 6): [-0.06666666666666654, 1.0666666666666664],
            (3, 1, 2): [-0.06666666666666654, 1.0666666666666664],
            (4, 1, 6): [-0.06666666666666654, 1.0666666666666664],
            (1, 1, 6): [-0.06666666666666654, 1.0666666666666664],
            (1, 2, 2): [0.022222222222222185, -0.444444444444444,
                        1.4222222222222216],
            (3, 2, 6): [0.0002614379084968331, -0.07111111111111235,
                        1.070849673202616],
            (1, 1, 4): [-0.06666666666666654, 1.0666666666666664],
            (2, 1, 2): [-0.06666666666666654, 1.0666666666666664],
            (4, 2, 4): [0.0002614379084968331, -0.07111111111111235,
                        1.070849673202616],
            (3, 2, 4): [0.0002614379084968331, -0.07111111111111235,
                        1.070849673202616],
            (2, 2, 2): [0.0002614379084968331, -0.07111111111111235,
                        1.070849673202616],
            (1, 2, 6): [0.0002614379084968331, -0.07111111111111235,
                        1.070849673202616],
            (4, 2, 6): [0.0002614379084968331, -0.07111111111111235,
                        1.070849673202616],
            (1, 1, 2): [-0.33333333333333304, 1.333333333333333],
            (3, 2, 8): [9.576480164718605e-07, -0.004167684167715291,
                        1.004166726519699],
            (2, 2, 6): [0.0002614379084968331, -0.07111111111111235,
                        1.070849673202616]}
        # t = dict()
        for n in [1, 2, 3, 4]:
            for num_terms in [1, 2]:
                for order in range(2, 9, 2):
                    d = nd.Derivative(np.exp, n=n, method='complex',
                                      order=order)
                    d._set_richardson_rule(step_ratio=2.0, num_terms=num_terms)
                    rule = d._richardson_extrapolate._get_richardson_rule()
                    # t[(n, num_terms, order)] = rule.tolist()
                    assert_array_almost_equal(rule,
                                              truth[(n, num_terms, order)])
        # print(t)
        # self.assert_(False)

    def test_forward_backward(self):
        truth = {(1, 1): [-1., 2.],
                 (1, 2): [-0.33333333, 1.33333333],
                 (1, 3): [-0.14285714, 1.14285714],
                 (1, 4): [-0.06666667, 1.06666667],
                 (1, 5): [-0.03225806, 1.03225806],
                 (1, 6): [-0.01587302, 1.01587302],
                 (2, 1): [0.33333333, -2., 2.66666667],
                 (2, 2): [0.04761905, -0.57142857, 1.52380952],
                 (2, 3): [0.00952381, -0.22857143, 1.21904762],
                 (2, 4): [0.00215054, -0.10322581, 1.10107527],
                 (2, 5): [5.12032770e-04, -4.91551459e-02, 1.04864311e+00],
                 (2, 6): [1.24984377e-04, -2.39970004e-02, 1.02387202e+00]}
        for method in ['forward', 'backward']:
            for num_terms in [1, 2]:
                for order in range(1, 7):
                    d = nd.Derivative(np.exp, method=method, order=order)
                    d._set_richardson_rule(step_ratio=2.0, num_terms=num_terms)
                    rule = d._richardson_extrapolate._get_richardson_rule()
                    assert_array_almost_equal(rule,
                                              truth[(num_terms, order)])

    def _example_(self):
        def f(x, h):
            return (np.exp(x + h) - np.exp(x - h)) / (2.)
        # f = lambda x, h: (np.exp(x+h)-np.exp(x))
        steps = [h for h in 2.0**-np.arange(10)]
        df = [f(1, h) for h in steps]
        print([dfi / hi for dfi, hi in zip(df, steps)])
        step = nd.MaxStepGenerator(step_ratio=2.0)
        for method in ['central']:
            d = nd.Derivative(np.exp, step=step, method=method)
            for order in [2, 6]:
                d.order = order
                r_extrap = nd.Richardson(step_ratio=2.0, method=method,
                                         num_terms=2, order=order)

                fd_rule = d._get_finite_difference_rule(step_ratio=2.0)
                print(fd_rule)
                df1, stepsi, _shape = d._apply_fd_rule(fd_rule, df, steps)

                rule = r_extrap._get_richardson_rule()
                df2, error, hi = r_extrap(df1, stepsi)

                print(rule)
                print(np.hstack((df2, error)))

        self.assert_(False)


class TestStepGenerator(unittest.TestCase):

    def test_default_generator(self):
        step_gen = nd.MinStepGenerator(base_step=None, num_steps=10,
                                       step_ratio=4, offset=-1)
        h = np.array([h for h in step_gen(0)])
        desired = np.array([3.58968236e-02, 8.97420590e-03, 2.24355147e-03,
                            5.60887869e-04, 1.40221967e-04, 3.50554918e-05,
                            8.76387295e-06, 2.19096824e-06, 5.47742059e-07,
                            1.36935515e-07])

        assert_array_almost_equal((h - desired) / desired, 0)

    def test_default_base_step(self):
        step_gen = nd.MinStepGenerator(num_steps=1, offset=0)
        h = [h for h in step_gen(0)]
        desired = nd.EPS ** (1. / 2.5)
        assert_array_almost_equal((h[0] - desired) / desired, 0)

    def test_fixed_base_step(self):
        desired = 0.1
        step_gen = nd.MinStepGenerator(base_step=desired, num_steps=1, scale=2,
                                       offset=0)
        h = [h for h in step_gen(0)]
        assert_array_almost_equal((h[0] - desired) / desired, 0)


class TestFornbergWeights(unittest.TestCase):

    def test_weights(self):
        x = np.r_[-1, 0, 1]
        xbar = 0
        k = 1
        weights = nd.fornberg_weights(x, xbar, k)
        np.testing.assert_allclose(weights, [-.5, 0, .5])


class TestDerivative(unittest.TestCase):

    def test_infinite_functions(self):
        def finf(x):
            return np.inf
        df = nd.Derivative(finf)
        val = df(0)
        self.assert_(np.isnan(val))

    def _example_fd_mat(self):
        fdmat = nd.Derivative._fd_matrix(step_ratio=2.0, parity=1, nterms=3)
        _fd_rules = np.linalg.pinv(fdmat)
        self.assert_(False)

    def test_high_order_derivative_cos(self):
        true_vals = (-1.0, 0.0, 1.0, 0.0, -1.0, 0.0)
        methods = ['complex', 'multicomplex', 'central',
                   'forward', 'backward']
        for method in methods:
            n_max = dict(multicomplex=2, central=6).get(method, 5)
            for n in range(1, n_max + 1):
                true_val = true_vals[n - 1]
                for order in range(2, 9, 2):
                    d3cos = nd.Derivative(np.cos, n=n, order=order,
                                          method=method, full_output=True)
                    y, info = d3cos(np.pi / 2.0)
                    error = np.abs(y - true_val)
                    small = error <= info.error_estimate
                    if not small:
                        small = error < 10**(-12 + n)
                    if not small:
                        print('method=%s, n=%d, order=%d' % (method, n, order))
                        print(error, info.error_estimate)
                    # self.assertTrue(small)
                    assert_array_almost_equal(y, true_val, decimal=4)
        # self.assert_(False)

    def test_derivative_of_cos_x(self):
        x = np.r_[0, np.pi / 6.0, np.pi / 2.0]
        true_vals = (-np.sin(x), -np.cos(x), np.sin(x), np.cos(x), -np.sin(x),
                     -np.cos(x))
        for method in ['complex', 'central', 'forward', 'backward']:
            n_max = dict(complex=2, central=6).get(method, 5)
            for n in range(1, n_max + 1):
                true_val = true_vals[n - 1]
                start, stop, step = dict(central=(2, 7, 2),
                                         complex=(2, 3, 1)).get(method,
                                                                (1, 5, 1))
                for order in range(start, stop, step):
                    d3cos = nd.Derivative(np.cos, n=n, order=order,
                                          method=method, full_output=True)
                    y, info = d3cos(x)
                    error = np.abs(y - true_val)
                    small = error <= info.error_estimate
                    if not small.all():
                        small = np.where(small, small, error <= 10**(-11 + n))
                    if not small.all():
                        print('method=%s, n=%d, order=%d' % (method, n, order))
                        print(error, info.error_estimate)
                    assert_array_almost_equal(y, true_val, decimal=4)
                    # self.assertTrue(small.all())
                    # assert_allclose(y, true_val)
        # self.assert_(False)

    def test_default_scale(self):
        for method, scale in zip(['complex', 'central', 'forward', 'backward',
                                  'multicomplex'],
                                 [1.35, 2.5, 2.5, 2.5, 1.35]):
            np.testing.assert_allclose(scale, nd.default_scale(method, n=1))

    def test_fun_with_additional_parameters(self):
        '''Test for issue #9'''
        def func(x, a, b=1):
            return b * a * x * x * x
        methods = ['forward', 'backward', 'central', 'complex', 'multicomplex']
        dfuns = [nd.Gradient, nd.Derivative,  nd.Jacobian, nd.Hessdiag,
                 nd.Hessian]
        for dfun in dfuns:
            for method in methods:
                df = dfun(func, method=method)
                val = df(0.0, 1.0, b=2)

                assert_array_almost_equal(val, 0)

    def test_derivative_cube(self):
        '''Test for Issue 7'''
        def cube(x):
            return x * x * x
        dcube = nd.Derivative(cube)
        shape = (3, 2)
        x = np.ones(shape) * 2
        dx = dcube(x)
        assert_array_almost_equal(list(dx.shape), list(shape),
                                  decimal=8,
                                  err_msg='Shape mismatch')
        txt = 'First differing element %d\n value = %g,\n true value = %g'
        for i, (val, tval) in enumerate(zip(dx.ravel(), (3 * x**2).ravel())):
            assert_array_almost_equal(val, tval, decimal=8,
                                      err_msg=txt % (i, val, tval))

    def test_derivative_exp(self):
        # derivative of exp(x), at x == 0
        dexp = nd.Derivative(np.exp)
        assert_array_almost_equal(dexp(0), np.exp(0), decimal=8)

    def test_derivative_sin(self):
        # Evaluate the indicated (default = first)
        # derivative at multiple points
        dsin = nd.Derivative(np.sin)
        x = np.linspace(0, 2. * np.pi, 13)
        y = dsin(x)
        np.testing.assert_almost_equal(y, np.cos(x), decimal=8)

    def test_backward_derivative_on_sinh(self):
        # Compute the derivative of a function using a backward difference
        # scheme.  A backward scheme will only look below x0.
        dsinh = nd.Derivative(np.sinh, method='backward')
        self.assertAlmostEqual(dsinh(0.0), np.cosh(0.0))

    def test_central_and_forward_derivative_on_log(self):
        # Although a central rule may put some samples in the wrong places, it
        # may still succeed
        epsilon = nd.MinStepGenerator(num_steps=15, offset=0, step_ratio=2)
        dlog = nd.Derivative(np.log, method='central', step=epsilon)
        x = 0.01
        self.assertAlmostEqual(dlog(x), 1.0 / x)

        # But forcing the use of a one-sided rule may be smart anyway
        dlog = nd.Derivative(np.log, method='forward', step=epsilon)
        self.assertAlmostEqual(dlog(x), 1 / x)


class TestJacobian(unittest.TestCase):

    def testjacobian(self):
        xdata = np.reshape(np.arange(0, 1, 0.1), (-1, 1))
        ydata = 1 + 2 * np.exp(0.75 * xdata)

        def fun(c):
            return (c[0] + c[1] * np.exp(c[2] * xdata) - ydata) ** 2

        for method in ['complex', 'central', 'forward', 'backward']:
            for order in [2, 4]:
                Jfun = nd.Jacobian(fun, method=method, order=order)
                J = Jfun([1, 2, 0.75])  # should be numerically zero
                assert_array_almost_equal(J, np.zeros(J.shape))


class TestGradient(unittest.TestCase):

    def testgradient(self):
        def fun(x):
            return np.sum(x ** 2)
        dtrue = [2., 4., 6.]

        for method in ['complex', 'central', 'backward', 'forward']:
            for order in [2, 4]:
                dfun = nd.Gradient(fun, method=method, order=order)
                d = dfun([1, 2, 3])
                assert_array_almost_equal(d, dtrue)
        # self.assert_(False)


class TestHessdiag(unittest.TestCase):

    def test_complex(self):
        def fun(x):
            return x[0] + x[1] ** 2 + x[2] ** 3
        htrue = np.array([0., 2., 18.])
        method = 'complex'
        for num_steps in range(3, 7, 1):
            steps = nd.MinStepGenerator(num_steps=num_steps,
                                        use_exact_steps=True,
                                        step_ratio=2.0, offset=4)
            Hfun = nd.Hessdiag(fun, step=steps, method=method,
                               full_output=True)
            hd, _info = Hfun([1, 2, 3])
            _error = hd - htrue
            assert_array_almost_equal(hd, htrue)

    def test_fixed_step(self):
        def fun(x):
            return x[0] + x[1] ** 2 + x[2] ** 3
        htrue = np.array([0., 2., 18.])

        methods = ['multicomplex', 'complex', 'central', 'forward', 'backward']
        for order in range(2, 7, 2):
            steps = nd.MinStepGenerator(num_steps=order + 1,
                                        use_exact_steps=True,
                                        step_ratio=3., offset=0)
            for method in methods:
                Hfun = nd.Hessdiag(fun, step=steps, method=method, order=order,
                                   full_output=True)
                hd, _info = Hfun([1, 2, 3])
                _error = hd - htrue
                assert_array_almost_equal(hd, htrue)

    def test_default_step(self):
        def fun(x):
            return x[0] + x[1] ** 2 + x[2] ** 3
        htrue = np.array([0., 2., 18.])
        methods = ['central2', 'central', 'multicomplex', 'complex', 'forward',
                   'backward']
        for order in range(2, 7, 2):
            for method in methods:
                Hfun = nd.Hessdiag(fun, method=method, order=order,
                                   full_output=True)
                hd, _info = Hfun([1, 2, 3])
                _error = hd - htrue
                assert_array_almost_equal(hd, htrue)


class TestHessian(unittest.TestCase):

    def test_hessian_cosIx_yI_at_I0_0I(self):
        # cos(x-y), at (0,0)

        def fun(xy):
            return np.cos(xy[0] - xy[1])
        htrue = [[-1., 1.], [1., -1.]]
        methods = ['multicomplex', 'complex', 'central', 'central2', 'forward',
                   'backward']
        for num_steps in [10, 1]:
            step = nd.MinStepGenerator(num_steps=num_steps)
            for method in methods:
                Hfun2 = nd.Hessian(fun, method=method, step=step,
                                   full_output=True)
                h2, _info = Hfun2([0, 0])
                # print(method, (h2-np.array(htrue)))
                assert_array_almost_equal(h2, htrue)


if __name__ == '__main__':
    unittest.main()
