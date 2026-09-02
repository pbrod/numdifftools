import pytest

profiletools = pytest.importorskip("profiletools")

from numdifftools.profile_numdifftools import main, profile_hessian  # noqa: E402


def test_profile_numdifftools_main() -> None:
    main()


def test_profile_numdifftools_profile_hessian() -> None:
    profile_hessian()
