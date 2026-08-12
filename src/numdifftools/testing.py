"""
Created on Apr 4, 2016

@author: pab
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Generator, Sequence
from io import StringIO
from timeit import default_timer as timer
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from numdifftools._typing import ArrayOrScalar


def rosen(x: ArrayLike) -> ArrayOrScalar:
    """Rosenbrock function

    This is a non-convex function used as a performance test problem for
    optimization algorithms introduced by Howard H. Rosenbrock in 1960.[1]
    """
    x = np.atleast_1d(x)
    return (1 - x[0]) ** 2 + 105.0 * (x[1] - x[0] ** 2) ** 2


def test_docstrings(filename: str | None = None) -> Any:
    import doctest

    if filename:
        print(f"Running doctests in {filename}...")
    else:
        print("Running doctests...")

    t0 = timer()
    result = doctest.testmod(optionflags=(doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS))
    dt = timer() - t0

    print(f"Attempted: {result.attempted}, Failed: {result.failed}, Elapsed: {dt:.3f}s")
    return result


def test(
    package_name: str,
    *options: str,
    plugins: Any | None = None,
) -> int:
    """
    Run tests for package using pytest.

    Parameters
    ----------
    package_name : str
        The name of the package to test.
    *options : optional
        options to pass to pytest. The most important ones include:
        '-v', '--verbose':
            increase verbosity.
        '-q', '--quiet':
            decrease verbosity.
        '--doctest-modules':
            run doctests in all .py modules
        '--cov':
            measure coverage for .py modules (requires pytest-cov plugin)
        '-h', '--help':
            show full help message and display all possible options to use.

    Returns
    -------
    exit_code: int
        Exit code is 0 if all tests passed without failure.

    Examples
    --------
    {super}

    """
    try:
        import pytest
    except ImportError as exc:
        raise ImportError(
            "pytest is required to run package tests. Install it with: pip install pytest."
        ) from exc

    return pytest.main(
        ["--pyargs", package_name, *options],
        plugins=plugins,
    )


@contextlib.contextmanager
def capture_stdout_and_stderr() -> Generator[Sequence[StringIO | str], None, None]:
    """
    Capture sys.stdout and sys.stderr

    Examples
    --------
    >>> from numdifftools.testing import capture_stdout_and_stderr
    >>> with capture_stdout_and_stderr() as out:
    ...    print('This is a test')
    >>> out[0].startswith('This is a test')
    True
    >>> out[1] == ''
    True
    """
    old_out = sys.stdout, sys.stderr
    out: list[StringIO | str]
    out = [StringIO(), StringIO()]
    try:
        sys.stdout, sys.stderr = out
        yield out
    finally:
        sys.stdout, sys.stderr = old_out
        if isinstance(out[0], StringIO):
            out[0] = out[0].getvalue()
        if isinstance(out[1], StringIO):
            out[1] = out[1].getvalue()


if __name__ == "__main__":
    test_docstrings(__file__)
