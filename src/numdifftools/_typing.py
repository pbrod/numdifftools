"""


MinStepGenerator      \
MaxStepGenerator       > StepGeneratorFactory
CStepGenerator        /

            |
            v

BasicMinStepGenerator \
BasicMaxStepGenerator  > StepGenerator


"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any, NamedTuple, Protocol, TypeAlias

from numpy.typing import ArrayLike, NDArray

Array: TypeAlias = NDArray[Any]
Scalar: TypeAlias = float | complex
ArrayOrScalar: TypeAlias = Array | Scalar
GeneratorStepRatio: TypeAlias = float | complex
RuleClass: TypeAlias = type[Any]

FunctionLike: TypeAlias = Callable[..., Any]
MathFunc: TypeAlias = Callable[..., ArrayOrScalar]
FuncOrNone: TypeAlias = MathFunc | None
FunctionPair: TypeAlias = tuple[
    FuncOrNone,
    FuncOrNone,
]
DerivativeFactory = Callable[[int], FuncOrNone]


class EstimateResult(NamedTuple):
    estimate: ArrayOrScalar
    error_estimate: ArrayOrScalar
    final_step: ArrayOrScalar
    best_index: int | Array


class ExtrapolatedSequence(NamedTuple):
    values: Array
    error_estimate: Array
    steps: Array


class DerivativeCallable(Protocol):
    def __call__(
        self,
        x: ArrayLike,
        f: FunctionLike,
        epsilon: StepGeneratorFactory | ArrayLike | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Array: ...


class DifferenceFunction(Protocol):
    def __call__(
        self,
        fun: Callable[..., Any],
        f_xi: Any,
        x_i: Array,
        h: Array,
    ) -> Array | ArrayOrScalar: ...


class Differentiator(Protocol):
    fun: Any

    def __call__(
        self,
        x: ArrayLike,
        *args: Any,
        **kwargs: Any,
    ) -> ArrayOrScalar | EstimateResult: ...


class FiniteDifferenceRule(Protocol):
    """Protocol implemented by LogRule and friends."""

    n: int
    order: int
    method: str
    method_order: int
    richardson_step: int
    eval_first_condition: bool

    def diff(
        self,
        fun: Callable[..., Any],
        f_xi: Any,
        x_i: Array,
        h: Array,
    ) -> Any: ...

    def apply(
        self,
        sequence: Sequence[ArrayOrScalar],
        steps: Sequence[ArrayOrScalar],
        step_ratio: float,
    ) -> tuple[Array, Array, tuple[int, ...]]: ...


class StepGenerator(Protocol):
    """Concrete generator returned by step_generator_function().

    Notes
    ----
    Generator is either BasicMinStepGenerator or BasicMinStepGenerator.
    """

    step_ratio: GeneratorStepRatio
    extrapolation_ratio: float

    def __call__(self) -> Iterator[ArrayOrScalar]: ...


class StepGeneratorFactory(Protocol):
    """Implemented by MinStepGenerator and MaxStepGenerator."""

    # step_ratio: float | None
    scale: float | None

    @property
    def step_ratio(self) -> float: ...

    @step_ratio.setter
    def step_ratio(self, value: float | None) -> None: ...

    def step_generator_function(
        self,
        x: ArrayLike,
        method: str,
        n: int,
        order: int,
    ) -> StepGenerator: ...

    def __call__(
        self,
        x: ArrayLike,
        method: str = "forward",
        n: int = 1,
        order: int = 2,
    ) -> Iterator[ArrayOrScalar]: ...


StepArgument: TypeAlias = ArrayLike | StepGeneratorFactory | None


class RichardsonLike(Protocol):
    """Structural interface for Richardson extrapolation."""

    step_ratio: float
    step: int | float
    order: int
    num_terms: int

    def __call__(
        self,
        sequence: Array,
        steps: Array,
    ) -> ExtrapolatedSequence: ...
