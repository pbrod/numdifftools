"""
This script profile different parts of numdifftools.

"""

from collections.abc import Callable, Iterable
from typing import Any, cast

import numpy as np
from profiletools import do_cprofile, do_profile

import numdifftools as nd  # numdifftools.nd_statsmodels as nd
from numdifftools._typing import (
    ArrayOrScalar,
    EstimateResult,
    FuncOrNone,
    MathFunc,
)
from numdifftools.example_functions import function_names, get_function
from numdifftools.run_benchmark import BenchmarkFunction


def profile_hessian(
    n_values: Iterable[int] = (4, 8, 16, 32, 64, 96),
) -> None:
    for n in n_values:
        f: MathFunc = BenchmarkFunction(n)

        step = nd.step_generators.one_step
        cls = cast(Any, nd.Hessian(f, step=step, method="central"))
        # pylint: disable=protected-access
        fd_rule = cls._fd_rule

        difference_functions = fd_rule._difference_functions
        follow: tuple[Callable[..., Any], ...] = (
            cls._derivative_nonzero_order,
            fd_rule.apply,
            fd_rule._prepare_extrapolation_data,
            difference_functions._central_even,
        )

        #         cls = nds.Hessian(f, step=None, method='central')
        #         follow = (cls._derivative_nonzero_order, )

        x = 3 * np.ones(n)

        do_profile(follow=follow)(cls)(x)


@do_cprofile  # type: ignore[untyped-decorator]
def main() -> None:
    x: float = 0.5
    methods: list[str] = ["complex", "central", "backward", "forward"]

    f: FuncOrNone
    true_df: FuncOrNone

    # for i, derivative in enumerate([nd.Derivative, nds.Gradient, nda.Derivative]):
    i = 0
    derivative = nd.Derivative
    for name in function_names:
        if i > 1 and name in ["arcsinh", "exp2"]:
            continue

        f, true_df = get_function(name, n=1)
        if true_df is None or f is None:
            continue
        assert true_df is not None
        assert f is not None
        for method in methods[3 * (i > 1) :]:
            df = derivative(f, method=method)
            result: ArrayOrScalar | EstimateResult = df(x)
            val: ArrayOrScalar = result.estimate if isinstance(result, EstimateResult) else result
            tval = true_df(x)

            dm = 7
            print(i, name, method, dm, np.abs(val - tval))


def profile_main() -> None:
    import cProfile
    import pstats

    cProfile.run("main()", f"{__file__}.profile")
    s = pstats.Stats(f"{__file__}.profile")

    s.sort_stats("time").print_stats(20)


if __name__ == "__main__":
    # profile_hessian()
    # profile_main()
    main()
