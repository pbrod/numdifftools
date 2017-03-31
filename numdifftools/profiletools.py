from __future__ import print_function
import inspect
import time
import cProfile
from functools import wraps

try:
    from line_profiler import LineProfiler

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
        Just decorate your test function or class method and pass any additional problem
        function(s) in the follow argument! If any follow argument is a string, it is
        assumed that the string refers to bound a method of the class

        See also
        --------
        do_cprofile, test_do_profile
        """
        def inner(func):

            def add_all_class_methods(profiler, cls):
                for k, v in inspect.getmembers(cls, inspect.ismethod):
                    if k != func.__name__:
                        profiler.add_function(v)

            def profiled_func(*args, **kwargs):
                try:
                    profiler = LineProfiler()
                    profiler.add_function(func)
                    if follow_all_methods:
                        cls = args[0]  # class instance
                        add_all_class_methods(profiler, cls)
                    for f in follow:
                        if isinstance(f, str):  # f is a method of the
                            cls = args[0]  # class instance
                            profiler.add_function(getattr(cls, f))
                        else:
                            profiler.add_function(f)
                    profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    profiler.print_stats()
            return profiled_func
        return inner

except ImportError:
    def do_profile(follow=()):
        "Helpful if you accidentally leave in production!"
        def inner(func):
            def nothing(*args, **kwargs):
                return func(*args, **kwargs)
            return nothing
        return inner


def timefun(fun):
    @wraps(fun)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fun(*args, **kwargs)
        t2 = time.time()
        print("@timefun:" + fun.func_name + " took " + str(t2 - t1) + " seconds")
        return result
    return measure_time


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
    No external dependencies and quite fast. Useful for quick high-level checks.

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


def _get_number():
    for x in range(50000):
        yield x


def test_do_profile():
    """
    If you run this, you should see a report that looks something like this:

    Timer unit: 3.94754e-07 s

    Total time: 0.0352105 s
    File: C:\pab\workspace\git_numdifftools\numdifftools\testing.py
    Function: _get_number at line 112

    Line #      Hits         Time  Per Hit   % Time  Line Contents
    ==============================================================
       112                                           def _get_number():
       113     50001        47560      1.0     53.3      for x in xrange(50000):
       114     50000        41636      0.8     46.7          yield x

    Total time: 0.127297 s
    File: C:\pab\workspace\git_numdifftools\numdifftools\testing.py
    Function: expensive_function at line 146

    Line #      Hits         Time  Per Hit   % Time  Line Contents
    ==============================================================
       146                                               @do_profile(follow=[_get_number])
       147                                               def expensive_function():
       148     50001       263360      5.3     81.7          for x in _get_number():
       149     50000        59110      1.2     18.3              i = x ^ x ^ x
       150         1            1      1.0      0.0          return i

    Timer unit: 3.94754e-07 s

    Total time: 0.0350498 s
    File: C:\pab\workspace\git_numdifftools\numdifftools\testing.py
    Function: _get_number at line 112

    Line #      Hits         Time  Per Hit   % Time  Line Contents
    ==============================================================
       112                                           def _get_number():
       113     50001        45240      0.9     51.0      for x in xrange(50000):
       114     50000        43549      0.9     49.0          yield x

    Total time: 0.130155 s
    File: C:\pab\workspace\git_numdifftools\numdifftools\testing.py
    Function: expensive_method1 at line 156

    Line #      Hits         Time  Per Hit   % Time  Line Contents
    ==============================================================
       156                                                   @do_profile(follow=[_get_number])
       157                                                   def expensive_method1(self):
       158     50001       266317      5.3     80.8              for x in _get_number():
       159     50000        63393      1.3     19.2                  i = x ^ x ^ x ^ x
       160         1            1      1.0      0.0              return i

    Timer unit: 3.94754e-07 s

    Total time: 0.00330172 s
    File: C:\pab\workspace\git_numdifftools\numdifftools\testing.py
    Function: _get_number at line 162

    Line #      Hits         Time  Per Hit   % Time  Line Contents
    ==============================================================
       162                                                   def _get_number(self):
       163      5001         4312      0.9     51.6              for x in xrange(5000):
       164      5000         4052      0.8     48.4                  yield x

    Total time: 0.0118632 s
    File: C:\pab\workspace\git_numdifftools\numdifftools\testing.py
    Function: expensive_method2 at line 166

    Line #      Hits         Time  Per Hit   % Time  Line Contents
    ==============================================================
       166                                                   def expensive_method2(self):
       167      5001        24509      4.9     81.6              for x in self._get_number():
       168      5000         5542      1.1     18.4                  i = x ^ x ^ x ^ x
       169         1            1      1.0      0.0              return i

    """

    @do_profile(follow=[_get_number])
    def expensive_function():
        for x in _get_number():
            i = x ^ x ^ x
        return i


    class ExpensiveClass1(object):
        @do_profile(follow=[_get_number])
        def expensive_method1(self):
            for x in _get_number():
                i = x ^ x ^ x ^ x
            return i

    class ExpensiveClass2(object):
        # You can not put class method _get_number2 into follow
        #
        @do_profile(follow=['_get_number2'])
        def expensive_method2(self):
            for x in self._get_number2():
                i = x ^ x ^ x ^ x
            return i

        def _get_number2(self):
            for x in xrange(5000):
                yield x

    # Profile all methods of ExpensiveClass3

    class ExpensiveClass3(object):
        @do_profile(follow_all_methods=True)
        def expensive_method3(self):
            for x in self._get_number3():
                for y in self._get_number32():
                    i = x ^ x ^ x ^ y
            return i

        def _get_number3(self):
            for x in range(5000):
                yield x

        def _get_number32(self):
            for x in range(50):
                yield x


    class ExpensiveClass4(object):
        def expensive_method4(self):
            for x in self._get_number4():
                i = x ^ x ^ x
            return i

        def _get_number4(self):
            for x in range(5000):
                yield x

    _test0 = expensive_function()
    _test1 = ExpensiveClass1().expensive_method1()
    _test2 = ExpensiveClass2().expensive_method2()
    _test3 = ExpensiveClass3().expensive_method3()
    cls = ExpensiveClass4()
    _test4_2 = do_profile(follow=[cls._get_number4])(cls.expensive_method4)()


def test_do_cprofile():
    """
    If you run this, you should see a report that looks something like this:

           5000003 function calls in 1.184 seconds

     Ordered by: standard name

     ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    5000001    0.326    0.000    0.326    0.000 testing.py:118(_get_number)
          1    0.858    0.858    1.184    1.184 testing.py:163(expensive_function2)
          1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}

    """
    @do_cprofile
    def expensive_function2():
        for x in _get_number():
            i = x ^ x ^ x ^ x
        return 'some result!'

    result2 = expensive_function2()


if __name__ == '__main__':
    test_do_profile()
    # test_do_cprofile()
