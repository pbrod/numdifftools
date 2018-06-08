import pytest
from numdifftools.profile_numdifftools import main, profile_hessian
from numdifftools._find_default_scale import run_all_benchmarks
from numdifftools.run_benchmark import run_gradient_and_hessian_benchmarks


def test_profile_numdifftools_main():
    main()


def test_profile_numdifftools_profile_hessian():
    profile_hessian()


def test__find_default_scale_run_all_benchmarks():
    run_all_benchmarks(method='forward', order=2, x_values=[0.1,], n_max=3)


def test_run_gradient_and_hessian_benchmarks():
    run_gradient_and_hessian_benchmarks(problem_sizes=(4, 8, 16))
