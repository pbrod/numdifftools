"""Test functions for numdifftools module"""
from __future__ import print_function
import unittest
import numdifftools.core as nd
from numdifftools.step_generators import default_scale
import numpy as np
from numpy.testing import assert_allclose
from numdifftools.testing import rosen
from numdifftools.tests.hamiltonian import run_hamiltonian
from hypothesis import given, example, note, strategies as st


class TestRichardson(unittest.TestCase):

    @staticmethod
    def test_central_forward_backward():
        central = {(1, 1): [-0.33333333, 1.33333333],
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
        forward = {(1, 1): [-1., 2.],
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
        true_vals = {'central': central, 'forward': forward,
                     'backward': forward}

        for method in true_vals:
            truth = true_vals[method]
            for num_terms in [1, 2]:
                for order in range(1, 7):
                    d = nd.Derivative(np.exp, method=method, order=order)
                    d.set_richardson_rule(step_ratio=2.0, num_terms=num_terms)
                    rule = d.richardson.rule()
                    assert_allclose(rule, truth[(num_terms, order)], atol=1e-8)

    @staticmethod
    def test_complex():
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

        for n in [1, 2, 3, 4]:
            for num_terms in [1, 2]:
                for order in range(2, 9, 2):
                    d = nd.Derivative(np.exp, n=n, method='complex',
                                      order=order)
                    d.set_richardson_rule(step_ratio=2.0, num_terms=num_terms)
                    rule = d.richardson.rule()

                    msg = "n={0}, num_terms={1}, order={2}".format(n,
                                                                   num_terms,
                                                                   order)
                    assert_allclose(rule,
                                    truth[(n, num_terms, order)],
                                    err_msg=msg)


class TestDerivative(unittest.TestCase):

    @staticmethod
    def test_directional_diff():
        v = [1, -1]
        x0 = [2, 3]

        directional_diff = nd.directionaldiff(rosen, x0, v)
        assert_allclose(directional_diff, 743.87633380824832)

    def test_infinite_functions(self):
        def finf(x):
            return np.inf * np.ones_like(x)
        df = nd.Derivative(finf)
        val = df(0)
        self.assert_(np.isnan(val))

        df.n = 0
        self.assertEqual(df(0), np.inf)

    @staticmethod
    def test_high_order_derivative_cos():
        true_vals = (0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0)
        methods = ['complex', 'multicomplex', 'central',
                   'forward', 'backward']
        for method in methods:
            n_max = dict(multicomplex=2, central=6).get(method, 5)
            for n in range(0, n_max + 1):
                true_val = true_vals[n]
                for order in range(2, 9, 2):
                    d3cos = nd.Derivative(np.cos, n=n, order=order,
                                          method=method, full_output=True)
                    y, _info = d3cos(np.pi / 2.0)
                    _error = np.abs(y - true_val)
#                 small = error <= info.error_estimate
#                 if not small:
#                     small = error < 10**(-12 + n)
#                 if not small:
#                     print('method=%s, n=%d, order=%d' % (method, n, order))
#                     print(error, info.error_estimate)
#                 self.assertTrue(small)
                    assert_allclose(y, true_val, atol=1e-4)
        # self.assert_(False)

    @staticmethod
    @given(st.floats(min_value=0, max_value=10))
    @example(7.6564547238847105)
    def test_derivative_of_cos_x(x):
        note('x = {}'.format(x))
        msg = 'order = {}, error = {}, err_est = {}'
        true_vals = (-np.sin(x), -np.cos(x), np.sin(x), np.cos(x)) * 2
        for method in ['complex', 'central', 'forward', 'backward']:
            note('method = {}'.format(method))
            n_max = dict(complex=7, central=6).get(method, 4)
            for n in range(1, n_max + 1):
                true_val = true_vals[n - 1]
                start, stop, step = dict(central=(2, 7, 2),
                                         complex=(2, 3, 1)).get(method,
                                                                (1, 5, 1))
                note('n = {}, true_val = {}'.format(n, true_val))
                for order in range(start, stop, step):
                    d3cos = nd.Derivative(np.cos, n=n, order=order,
                                          method=method, full_output=True)
                    y, _info = d3cos(x)
                    _error = np.abs(y - true_val)
                    aerr = 100 * _info.error_estimate + 1e-14
#                     if aerr < 1e-14 and np.abs(true_val) < 1e-3:
#                         aerr = 1e-8
                    note(msg.format(order, _error, _info.error_estimate))
                    assert_allclose(y, true_val, rtol=1e-6, atol=aerr)
                    # assert_allclose(y, true_val, rtol=4)

    @staticmethod
    def test_default_scale():
        for method, scale in zip(['complex', 'central', 'forward', 'backward',
                                  'multicomplex'],
                                 [1.35, 2.5, 2.5, 2.5, 1.35]):
            np.testing.assert_allclose(scale, default_scale(method, n=1))

    @staticmethod
    def test_fun_with_additional_parameters():
        """Test for issue #9"""
        def func(x, a, b=1):
            return b * a * x * x * x
        methods = ['forward', 'backward', 'central', 'complex', 'multicomplex']
        dfuns = [nd.Gradient, nd.Derivative, nd.Jacobian, nd.Hessdiag,
                 nd.Hessian]
        for dfun in dfuns:
            for method in methods:
                df = dfun(func, method=method)
                val = df(0.0, 1.0, b=2)
                assert_allclose(val, 0, atol=1e-14)

    @staticmethod
    def test_derivative_with_step_options():
        def func(x, a, b=1):
            return b * a * x * x * x
        methods = ['forward', 'backward', 'central', 'complex', 'multicomplex']
        dfuns = [nd.Gradient, nd.Derivative, nd.Jacobian, nd.Hessdiag,
                 nd.Hessian]
        step_options = dict(num_extrap=5)
        for dfun in dfuns:
            for method in methods:
                df = dfun(func, method=method, **step_options)
                val = df(0.0, 1.0, b=2)
                assert_allclose(val, 0, atol=1e-14)

    @staticmethod
    def test_derivative_cube():
        """Test for Issue 7"""
        def cube(x):
            return x * x * x
        dcube = nd.Derivative(cube)
        shape = (3, 2)
        x = np.ones(shape) * 2
        dx = dcube(x)
        assert_allclose(list(dx.shape), list(shape), err_msg='Shape mismatch')
        txt = 'First differing element %d\n value = %g,\n true value = %g'
        for i, (val, tval) in enumerate(zip(dx.ravel(), (3 * x**2).ravel())):
            assert_allclose(val, tval, rtol=1e-8, err_msg=txt % (i, val, tval))

    @staticmethod
    def test_derivative_exp():
        # derivative of exp(x), at x == 0
        dexp = nd.Derivative(np.exp)
        assert_allclose(dexp(0), np.exp(0), rtol=1e-8)

    @staticmethod
    def test_derivative_sin():
        # Evaluate the indicated (default = first)
        # derivative at multiple points
        dsin = nd.Derivative(np.sin)
        x = np.linspace(0, 2. * np.pi, 13)
        y = dsin(x)
        assert_allclose(y, np.cos(x), atol=1e-8)

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
        x = 0.001
        self.assertAlmostEqual(dlog(x), 1.0 / x)

        # But forcing the use of a one-sided rule may be smart anyway
        dlog = nd.Derivative(np.log, method='forward', step=epsilon)
        self.assertAlmostEqual(dlog(x), 1 / x)


class TestJacobian(unittest.TestCase):

    @staticmethod
    @given(st.floats(min_value=-1000, max_value=1000))
    def test_scalar_to_vector(val):
        def fun(x):
            return np.array([x, x**2, x**3])
        truth = np.array([[1., 2 * val, 3 * val**2]])
        for method in ['multicomplex', 'complex', 'central', 'forward',
                       'backward']:
            j0, info = nd.Jacobian(fun, method=method, full_output=True)(val)
            error = np.abs(j0-truth)
            note('method={}, error={}, error_est={}'.format(method, error,
                                                            info.error_estimate))
            assert_allclose(j0, truth, rtol=1e-3, atol=1e-6)

    @staticmethod
    def test_on_scalar_function():
        def fun(x):
            return x[0] * x[1] * x[2] + np.exp(x[0]) * x[1]

        for method in ['complex', 'central', 'forward', 'backward']:
            j_fun = nd.Jacobian(fun, method=method)
            x = j_fun([3., 5., 7.])
            assert_allclose(x, [[135.42768462, 41.08553692, 15.]])

    @staticmethod
    def test_on_vector_valued_function():
        xdata = np.arange(0, 1, 0.1) #.reshape((-1, 1))
        ydata = 1 + 2 * np.exp(0.75 * xdata)

        def fun(c):
            return (c[0] + c[1] * np.exp(c[2] * xdata) - ydata) ** 2

        _j_0 = nd.approx_fprime([1, 2, 0.75], fun)

        for method in ['complex', 'central', 'forward', 'backward']:
            for order in [2, 4]:
                j_fun = nd.Jacobian(fun, method=method, order=order)
                j_val = j_fun([1, 2, 0.75])  # should be numerically zero
                assert_allclose(j_val, np.zeros((ydata.size, 3)), atol=1e-12)

    @staticmethod
    def test_on_matrix_valued_function():
        def fun(x):
            x = np.atleast_1d(x)
            f0 = x[0] ** 2 + x[1] ** 2
            f1 = x[0] ** 3 + x[1] ** 3
            return np.array([f0, f1])

        def dfun(x):
            x = np.atleast_1d(x)
            f0_d0 = np.atleast_1d(x[0] * 2)
            f0_d1 = np.atleast_1d(x[1] * 2)
            f1_d0 = np.atleast_1d(3 * x[0] ** 2)
            f1_d1 = np.atleast_1d(3 * x[1] ** 2)
            # algopy way:
            # df0 = np.hstack([np.diag(f0_d0), np.diag(f0_d1)])
            # df1 = np.hstack([np.diag(f1_d0), np.diag(f1_d1)])
            # numdifftools way:
            df0 = np.vstack([f0_d0, f0_d1])
            df1 = np.vstack([f1_d0, f1_d1])

            return np.array([df0, df1]).squeeze()

        jaca = nd.Jacobian(fun)
        assert_allclose(jaca([1, 2]), [[2., 4.],
                                       [3., 12.]])
        assert_allclose(jaca([3, 4]), [[6., 8.],
                                       [27., 48.]])
        assert_allclose(jaca([1, 2]), dfun([1, 2]))
        assert_allclose(jaca([3, 4]), dfun([3, 4]))

        v0 = jaca([[1, 2], [3, 4]])
        print(v0)
        assert_allclose(v0,
                        dfun([[1, 2],
                              [3, 4]]))

        x = np.array([(1, 2, 3, 4),
                      (5, 6, 7, 8)], dtype=float)

        y = fun(x)
        assert_allclose(y, [[26., 40., 58., 80.],
                            [126., 224., 370., 576.]])
        tval = dfun(x)
        assert_allclose(tval, [[[2., 4., 6., 8.],
                                [10., 12., 14., 16.]],
                               [[3., 12., 27., 48.],
                                [75., 108., 147., 192.]]])
        v0 = nd.approx_fprime(x, fun)
        val = jaca(x)
        assert_allclose(v0, tval)
        assert_allclose(val, tval)

    @staticmethod
    def test_issue_25():
        def g_fun(x):
            out = np.zeros((2, 2))
            out[0] = x
            out[1] = x
            return out

        dg_dx = nd.Jacobian(g_fun)
        x = np.array([1., 2.])
        v0 = nd.approx_fprime(x, g_fun)
        assert_allclose(v0, [[[1., 0.],
                             [0., 1.]],
                            [[1., 0.],
                             [0., 1.]]])

        dg = dg_dx(x)
        assert_allclose(dg, [[[1., 0.],
                             [0., 1.]],
                            [[1., 0.],
                             [0., 1.]]])
        fun3 = lambda x : np.vstack((x[0]*x[1]*x[2]**2, x[0]*x[1]*x[2]))
        Jfun3 = nd.Jacobian(fun3)
        x = np.array([[1.,2.,3.], [4., 5., 6.]]).T
        tv = [[[18., 180.],
               [9., 144.],
               [12., 240.]],
              [[6., 30.],
               [3., 24.],
               [2., 20.]]]
        assert_allclose(Jfun3(x), tv)
        assert_allclose(nd.approx_fprime(x, fun3), tv)


class TestGradient(unittest.TestCase):

    @staticmethod
    def test_directional_diff():
        v = np.r_[1, -1]
        v = v / np.linalg.norm(v)
        x0 = [2, 3]
        directional_diff = np.dot(nd.Gradient(rosen)(x0), v)
        assert_allclose(directional_diff, 743.87633380824832)
        dd, _info = nd.directionaldiff(rosen, x0, v, full_output=True)
        assert_allclose(dd, 743.87633380824832)

    @staticmethod
    def test_gradient():
        def fun(x):
            return np.sum(x ** 2)
        dtrue = [2., 4., 6.]

        for method in ['multicomplex', 'complex', 'central', 'backward',
                       'forward']:
            for order in [2, 4]:
                dfun = nd.Gradient(fun, method=method, order=order)
                d = dfun([1, 2, 3])
                assert_allclose(d, dtrue)
        # self.assert_(False)


class TestHessdiag(unittest.TestCase):

    @staticmethod
    def _fun(x):
        return x[0] + x[1] ** 2 + x[2] ** 3

    @staticmethod
    def _hfun(x):
        return 0, 2, 6 * x[2]

    def test_complex(self):

        htrue = np.array([0., 2., 18.])
        method = 'complex'
        for num_steps in range(3, 7, 1):
            steps = nd.MinStepGenerator(num_steps=num_steps,
                                        use_exact_steps=True,
                                        step_ratio=2.0, offset=4)
            h_fun = nd.Hessdiag(self._fun, step=steps, method=method,
                                full_output=True)
            h_val, _info = h_fun([1, 2, 3])
            _error = h_val - htrue
            assert_allclose(h_val, htrue)

    @given(st.tuples(st.floats(-100, 100), st.floats(-100, 100),
                     st.floats(-100, 100)) )
    @example((0.0, 16.003907623933742, 68.02057249966789))
    @example([1, 2, 3])
    @example((1.2009289063e-314, 1.2009289063e-314, 1.2009289063e-314))
    @example((88.01663712016305, 88.01663712016305, 88.01663712016305))
    def test_fixed_step(self, vals):
        htrue = self._hfun(vals)
        methods = ['central2', 'central', 'multicomplex', 'complex', 'forward',
                   'backward']
        for order in range(2, 7, 2):
            steps = nd.MinStepGenerator(num_steps=order + 1,
                                        use_exact_steps=True,
                                        step_ratio=3., offset=0)
            note('order = {}'.format(order))
            for method in methods:
                h_fun = nd.Hessdiag(self._fun, step=steps, method=method,
                                    order=order, full_output=True)
                h_val, _info = h_fun(vals)
                _error = np.abs(h_val - htrue)
                note('error = {}, error_est = {}'.format(_error,
                                                         _info.error_estimate))
                assert_allclose(h_val, htrue, atol=100*max(_info.error_estimate))

    def test_default_step(self):
        htrue = np.array([0., 2., 18.])
        methods = ['central2', 'central', 'multicomplex', 'complex', 'forward',
                   'backward']
        for order in range(2, 7, 2):
            for method in methods:
                h_fun = nd.Hessdiag(self._fun, method=method, order=order,
                                    full_output=True)
                h_val, _info = h_fun([1, 2, 3])
                _error = h_val - htrue
                tol = min(1e-8, _info.error_estimate.max())
                assert_allclose(h_val, htrue, atol=tol)


class TestHessian(unittest.TestCase):

    def test_run_hamiltonian(self):
        # Important to restrict the step in order to avoid the
        # discontinutiy at x=[0,0] of the hamiltonian

        for method in ['central', 'complex']:
            step = nd.MaxStepGenerator(base_step=1e-4)
            hessian = nd.Hessian(None, step=step, method=method)
            h, _error_estimate, true_h = run_hamiltonian(hessian,
                                                         verbose=False)
            self.assertTrue((np.abs((h - true_h) / true_h) < 1e-4).all())

    @staticmethod
    def test_hessian_cos_x_y_at_0_0():
        # cos(x-y), at (0,0)

        def fun(xy):
            return np.cos(xy[0] - xy[1])
        htrue = [[-1., 1.],
                 [1., -1.]]
        methods = ['multicomplex', 'complex', 'central', 'central2', 'forward',
                   'backward']
        for num_steps in [10, 1]:
            step = nd.MinStepGenerator(num_steps=num_steps)
            for method in methods:
                h_fun = nd.Hessian(fun, method=method, step=step,
                                   full_output=True)
                h_val, _info = h_fun([0, 0])
                # print(method, (h_val-np.array(htrue)))
                assert_allclose(h_val, htrue)


if __name__ == '__main__':
    unittest.main()
