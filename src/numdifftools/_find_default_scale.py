"""
This script can be run to find the empirical optimum scale for numdifftools.Derivative
given the method used.

Below are some results from previous runs compared with what is implemented in default_scale
function:

method="complex", order=2, x_values=[0.1, 0.5, 1.0, 5, 10, 50]:
n=1, mean scale=1.0330188679245282, median scale=1.0
n=2, mean scale=5.540094339622642, median scale=5.25
n=3, mean scale=8.36556603773585, median scale=7.625
n=4, mean scale=9.119791666666666, median scale=8.75
n=5, mean scale=10.154166666666667, median scale=9.625
n=6, mean scale=9.182291666666666, median scale=10.25
n=7, mean scale=14.078125, median scale=13.875
n=8, mean scale=14.307291666666666, median scale=14.875
n=9, mean scale=13.703125, median scale=13.625
n=10, mean scale=14.5625, median scale=14.75

Default scale with method="complex", order=2, x_values=[0.1, 0.5, 1.0, 5, 10, 50]:
n=1, scale=1.35
n=2, scale=5.0
n=3, scale=8.65
n=4, scale=11.35
n=5, scale=11.5
n=6, scale=11.7
n=7, scale=15.749999999999998
n=8, scale=21.35
n=9, scale=19.5
n=10, scale=20.78


method="central", order=2, x_values=0.5
n=1, scale=2.57894736842
n=2, scale=3.81578947368
n=3, scale=5.01315789474
n=4, scale=5.578125
n=5, scale=6.625
n=6, scale=7.59375
n=7, scale=8.65625
n=8, scale=9.28125
n=9, scale=9.84375

method="central", order=2, x_values=5
n=1, scale=2.86764705882
n=2, scale=5.41176470588
n=3, scale=6.23529411765
n=4, scale=5.859375
n=5, scale=8.025
n=6, scale=6.90625
n=7, scale=7.0625
n=8, scale=8.21875
n=9, scale=9.9375

method="central", order=2, x_values=[0.1, 0.5, 1.0, 5, 10, 50,]:
n=1, scale=2.77358490566
n=2, scale=4.75471698113
n=3, scale=5.19575471698
n=4, scale=5.7890625
n=5, scale=7.05
n=6, scale=7.046875
n=7, scale=7.89583333333
n=8, scale=8.41145833333
n=9, scale=9.21354166667
n=10, scale=9.33854166667

method="central", order=2, x_values=[0.1, 0.5, 1.0, 5, 10, 50]:
n=1, mean scale=2.7806603773584904, median scale=3.0
n=2, mean scale=4.75, median scale=4.0
n=3, mean scale=5.2334905660377355, median scale=4.75
n=4, mean scale=5.8046875, median scale=6.0
n=5, mean scale=7.05, median scale=6.5
n=6, mean scale=7.020833333333333, median scale=7.625
n=7, mean scale=7.458333333333333, median scale=7.75
n=8, mean scale=8.510416666666666, median scale=9.5
n=9, mean scale=8.328125, median scale=9.0
n=10, mean scale=9.265625, median scale=10.25

Default scale with method="central", order=2, x_values=[0.1, 0.5, 1.0, 5, 10, 50]:
n=1, scale=2.5
n=2, scale=3.8
n=3, scale=5.1
n=4, scale=6.4
n=5, scale=7.7
n=6, scale=9.0
n=7, scale=10.3
n=8, scale=11.6
n=9, scale=12.9
n=10, scale=14.200000000000001

"""

from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np

from numdifftools import Derivative
from numdifftools._typing import Array, ArrayOrScalar, EstimateResult, FunctionLike, StepGeneratorFactory
from numdifftools.example_functions import function_names, get_function
from numdifftools.step_generators import MinStepGenerator, default_scale


class BenchmarkResult(TypedDict):
    n: int
    order: int
    method: str
    fun: str
    error: float
    scale: float
    x: float


def plot_error(
    scales: Array,
    relative_error: Array,
    scale0: float,
    title: str = "",
    label: str = "",
) -> None:

    plt.semilogy(scales, relative_error, label=label)
    plt.vlines(scale0, np.nanmin(relative_error), 1)
    plt.xlabel("scales")
    plt.ylabel("Relative error")
    plt.title(title)
    plt.legend(frameon=False, framealpha=0.5)
    plt.axis(
        (
            float(min(scales)),
            float(max(scales)),
            float(np.nanmin(relative_error)),
            1.0,
        )
    )


def _compute_relative_errors(
    x: float,
    dfun: FunctionLike,
    fd: Derivative,
    scales: Array,
) -> Array:
    values: list[ArrayOrScalar] = []
    for scale in scales:
        fd.step.scale = scale
        try:
            val = fd(x)
            if isinstance(val, EstimateResult):
                val = val.estimate
        except Exception:
            val = np.nan

        values.append(val)

    t = np.array(values)
    tt = dfun(x)
    relativ_errors = np.abs(t - tt) / (np.maximum(np.abs(tt), 1)) + 1e-16
    return relativ_errors


def benchmark(
    x: float = 0.0001,
    dfun: FunctionLike | None = None,
    fd: Derivative | None = None,
    name: str = "",
    scales: Array | None = None,
    show_plot: bool = True,
) -> BenchmarkResult:
    if scales is None:
        scales = np.arange(1.0, 35, 0.25)

    assert fd is not None
    n, method, order = fd.n, fd.method, fd.order

    if dfun is None:
        return {
            "n": n,
            "order": order,
            "method": method,
            "fun": name,
            "error": np.nan,
            "scale": np.nan,
            "x": np.nan,
        }

    relativ_errors = _compute_relative_errors(x, dfun, fd, scales)

    if not np.isfinite(relativ_errors).any():
        return {
            "n": n,
            "order": order,
            "method": method,
            "fun": name,
            "error": np.nan,
            "scale": np.nan,
            "x": np.nan,
        }
    if show_plot:
        ordinal = ["", "1'st", "2'nd", "3'rd", "4'th", "5'th", "6'th", "7th"][n] if n < 8 else f"{n}'th"
        title = f"The {ordinal} derivative using {method}, order={order}"
        scale0 = default_scale(method, n, order)
        plot_error(scales, relativ_errors, scale0, title, label=name)

    i = np.nanargmin(relativ_errors)
    error = float(f"{relativ_errors[i]:.3g}")
    return {"n": n, "order": order, "method": method, "fun": name, "error": error, "scale": scales[i], "x": x}


def _print_summary(
    method: str,
    order: int,
    x_values: tuple[float, ...] | list[float],
    scales: dict[int, list[float]],
) -> None:
    print(scales)
    header = f'method="{method}", order={order}, x_values={str(x_values)}:'
    print(header)
    for n in scales:
        print(f"n={n}, mean scale={np.mean(scales[n]):.2f}, median scale={np.median(scales[n]):.2f}")

    print("Default scale with " + header)
    for n in scales:
        print(f"n={n}, scale={default_scale(method, n, order):.2f}")


def run_all_benchmarks(
    method: str = "forward",
    order: int = 4,
    x_values: tuple[float, ...] | list[float] = (0.1, 0.5, 1.0, 5),
    n_max: int = 11,
    show_plot: bool = True,
) -> None:
    epsilon: StepGeneratorFactory = MinStepGenerator(base_step=None, scale=None, step_nom=None, num_extrap=0)

    scales: dict[int, list[float]] = {}
    for n in range(1, n_max):
        plt.figure(n)
        scale_n = scales.setdefault(n, [])
        # for (name, x) in itertools.product( function_names, x_values):
        for name in function_names:
            fun0, dfun = get_function(name, n)
            if dfun is None:
                continue
            fd = Derivative(fun0, step=epsilon, method=method, n=n, order=order)
            for x in x_values:
                r = benchmark(x=x, dfun=dfun, fd=fd, name=name, scales=None, show_plot=show_plot)
                print(r)
                scale = r["scale"]
                if np.isfinite(scale):
                    scale_n.append(scale)

        plt.vlines(np.mean(scale_n), 1e-12, 1, "r", linewidth=3)
        plt.vlines(np.median(scale_n), 1e-12, 1, "b", linewidth=3)

    _print_summary(method, order, x_values, scales)


if __name__ == "__main__":
    run_all_benchmarks(
        method="complex",
        order=2,
        x_values=[0.1, 50],  # 0.1, 0.5, 1.0, 5, 10, 50,],
        n_max=11,
    )
    plt.show()
