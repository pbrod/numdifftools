"""
This module is based on: https://zapier.com/engineering/profiling-python-boss/

See also:
https://www.pythoncentral.io/measure-time-in-python-time-time-vs-time-clock/
"""

from __future__ import absolute_import, print_function
import inspect
import cProfile
from functools import wraps
from timeit import default_timer as timer
import warnings


try:
    from line_profiler import LineProfiler

    def _add_all_class_methods(profiler, cls, except_=''):
        for k, v in inspect.getmembers(cls, inspect.ismethod):
            if k != except_:
                profiler.add_function(v)

    def _add_function_or_classmethod(profiler, f, args):
        if isinstance(f, str):  # f is a method of the
            cls = args[0]  # class instance
            profiler.add_function(getattr(cls, f))
        else:
            profiler.add_function(f)

    def do_profile(follow=(), follow_all_methods=False):
        """
        Decorator to profile a function or class method

        It uses line_profiler to give detailed reports on time spent on each
        line in the code.

        Pros: has intuitive and finely detailed reports. Can follow
        functions in third party libraries.

        Cons:
        has external dependency on line_profiler and is quite slow,
        so don't use it for benchmarking.

        Handy tip:
        Just decorate your test function or class method and pass any
        additional problem function(s) in the follow argument!
        If any follow argument is a string, it is assumed that the string
        refers to bound a method of the class

        See also
        --------
        do_cprofile, test_do_profile
        """
        def inner(func):

            def profiled_func(*args, **kwargs):
                try:
                    profiler = LineProfiler()
                    profiler.add_function(func)
                    if follow_all_methods:
                        cls = args[0]  # class instance
                        _add_all_class_methods(profiler, cls,
                                               except_=func.__name__)
                    for f in follow:
                        _add_function_or_classmethod(profiler, f, args)
                    profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    profiler.print_stats()
            return profiled_func
        return inner

except ImportError as error:
    LineProfiler = None
    warnings.warn(str(error))

    def do_profile(follow=(), follow_all_methods=False):
        "Helpful if you accidentally leave in production!"
        def inner(func):
            def nothing(*args, **kwargs):
                return func(*args, **kwargs)
            return nothing
        return inner


def timefun(fun):
    """ Timing decorator

    Timers require you to do some digging. Start wrapping a few of the higher level
    functions and confirm where the bottleneck is, then drill down into that function,
    repeating as you go. When you find the disproportionately slow bit of code, fix it,
    and work your way back out confirming that it is fixed.

    Handy tip: Don't forget the handy timeit module! It tends to be more useful for
    benchmarking small pieces of code than for doing the actual investigation.

    Timer Pros:
    Easy to understand and implement. Also very simple to compare before and after fixes.
    Works across many languages.

    Timer Cons:
    Sometimes a little too simplistic for extremely complex codebases, you might spend
    more time placing and replacing boilerplate code than you will fixing the problem!

    """
    @wraps(fun)
    def measure_time(*args, **kwargs):
        t1 = timer()
        result = fun(*args, **kwargs)
        t2 = timer()
        print("@timefun:" + fun.__name__ + " took " + str(t2 - t1) + " seconds")
        return result
    return measure_time


class TimeWith():
    """
    Timing context manager

    """

    def __init__(self, name=''):
        self.name = name
        self.start = timer()

    @property
    def elapsed(self):
        return timer() - self.start

    def checkpoint(self, name=''):
        print('{timer} {checkpoint} took {elapsed} seconds'.format(timer=self.name,
                                                                   checkpoint=name,
                                                                   elapsed=self.elapsed,
                                                                   ).strip())

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.checkpoint('finished')


def do_cprofile(func):
    """
    Decorator to profile a function

    It gives good numbers on various function calls but it omits a vital piece
    of information: what is it about a function that makes it so slow?

    However, it is a great start to basic profiling. Sometimes it can even
    point you to the solution with very little fuss. I often use it as a
    gut check to start the debugging process before I dig deeper into the
    specific functions that are either slow are called way too often.

    Pros:
    No external dependencies and quite fast. Useful for quick high-level
    checks.

    Cons:
    Rather limited information that usually requires deeper debugging; reports
    are a bit unintuitive, especially for complex codebases.

    See also
    --------
    do_profile, test_do_profile
    """
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
    return profiled_func
