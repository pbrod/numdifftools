from __future__ import absolute_import, print_function
import pytest
import numdifftools.core as nd
import numpy as np
from numdifftools.step_generators import MinStepGenerator, MaxStepGenerator, EPS
from numpy.testing import assert_array_almost_equal, assert_equal


def test_min_step_generator_with_step_ratio4():
    step_gen = nd.MinStepGenerator(base_step=None, num_steps=10,
                                   step_ratio=4, offset=-1)
    h = np.array([h for h in step_gen(0)])
    desired = np.array([3.58968236e-02, 8.97420590e-03, 2.24355147e-03,
                        5.60887869e-04, 1.40221967e-04, 3.50554918e-05,
                        8.76387295e-06, 2.19096824e-06, 5.47742059e-07,
                        1.36935515e-07])

    assert_array_almost_equal((h - desired) / desired, 0)


def test_min_step_generator_default_base_step():
    step_gen = nd.MinStepGenerator(num_steps=1, offset=0)
    h = [h for h in step_gen(0)]
    desired = EPS ** (1. / 2.5)
    assert_array_almost_equal((h[-1] - desired) / desired, 0)


def test__min_step_generator_with_step_nom1():
    step_gen = nd.MinStepGenerator(num_steps=1, step_nom=1.0, offset=0)
    h = [h for h in step_gen(0)]
    desired = EPS ** (1. / 2.5)
    assert_array_almost_equal((h[-1] - desired) / desired, 0)


def test_min_step_generator_with_base_step01():
    desired = 0.1
    step_gen = nd.MinStepGenerator(base_step=desired, num_steps=1,
                                   offset=0)
    methods = ['forward', 'backward', 'central', 'complex']
    for n in range(1, 5):
        for order in [1, 2, 4, 6, 8]:
            min_length = n + order - 1
            lengths = [min_length, min_length, max(min_length // 2, 1),
                       max(min_length // 4, 1)]
            for m, method in zip(lengths, methods):
                h = [h for h in step_gen(0, method=method, n=n,
                                         order=order)]
                # print(len(h), n, order, method)
                assert_array_almost_equal((h[-1] - desired) / desired, 0)
                assert_equal(m, len(h))


def test_default_max_step_generator():
    step_gen = nd.MaxStepGenerator(num_steps=10)
    h = np.array([h for h in step_gen(0)])

    desired = 2.0 * 2.0 ** (-np.arange(10) + 0)

    assert_array_almost_equal((h - desired) / desired, 0)


def test_max_step_generator_default_base_step():
    step_gen = nd.MaxStepGenerator(num_steps=1, offset=0)
    h = [h for h in step_gen(0)]
    desired = 2.0
    assert_array_almost_equal((h[0] - desired) / desired, 0)


def test_max_step_generator_with_base_step01():
    desired = 0.1
    step_gen = nd.MaxStepGenerator(base_step=desired, num_steps=1, offset=0)
    methods = ['forward', 'backward', 'central', 'complex']
    lengths = [2, 2, 1, 1]
    for n, method in zip(lengths, methods):
        h = [h for h in step_gen(0, method=method)]
        assert_array_almost_equal((h[0] - desired) / desired, 0)
        assert_equal(n, len(h))
