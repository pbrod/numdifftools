from numdifftools import run_benchmark
from numdifftools._find_default_scale import run_all_benchmarks


def test__find_default_scale_run_all_benchmarks():
    run_all_benchmarks(
        method="forward",
        order=2,
        x_values=[
            0.1,
        ],
        n_max=3,
    )


def test_run_gradient_and_hessian_benchmarks():
    run_benchmark.main(problem_sizes=(4, 8, 16))
