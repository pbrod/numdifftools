from collections.abc import Callable
from typing import Any

import pytest

from numdifftools.profiletools import LineProfiler, TimeWith, do_cprofile, do_profile, timefun
from numdifftools.testing import capture_stdout_and_stderr


def _get_stats(line):
    item, _, tail = line.partition(" ")
    try:
        line_no = int(item)
    except ValueError:
        return None
    try:
        vals = [0] * 4
        for i in range(4):
            item, _, tail = tail.strip().partition(" ")
            vals[i] = float(item)

        hits, time, perhit, percent_time = vals
    except ValueError:
        tail = " ".join((item, tail))
        hits, time, perhit, percent_time = 0, 0, 0.0, 0.0
    return line_no, hits, time, perhit, percent_time, tail


def _extract_do_profile_results(txt, header_start="Line #"):
    results = []
    for line in txt.split("\n"):
        line = line.strip()
        if line.startswith(header_start):
            results.append(line)
            continue
        stats = _get_stats(line)
        if stats:
            results.append(stats)

    return results


def _extract_do_cprofile_results(txt):
    """
     ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    5000001    0.326    0.000    0.326    0.000 testing.py:118(_get_number)
          1    0.858    0.858    1.184    1.184 testing.py:163(expensive_function2)
          1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
    """
    return _extract_do_profile_results(txt, header_start="ncalls")


def _get_number():
    yield from range(50000)


def _expensive_function():
    i = 0
    for x in _get_number():
        i = i ^ x
    return i


class ExpensiveClass4:
    n = 5000

    def expensive_method4(self):
        for x in self._get_number4():
            i = x**3
        return i

    def _get_number4(self):
        yield from range(self.n)


FIRST_LINE = "Line #      Hits         Time  Per Hit   % Time  Line Contents"


# Tests
class TestTimeFun:
    def test_decorate_function(self):
        @timefun
        def expensive_function():
            for x in _get_number():
                i = x**3
            return i

        with capture_stdout_and_stderr() as out:
            expensive_function()
        msg = str(out)
        # print(out)
        assert len(out), msg
        assert out[0].startswith("@timefun:expensive_function took"), msg
        time = float(out[0].split("took")[1].strip().split(" ")[0])
        assert time > 0

    def test_direct_on_function(self):
        with capture_stdout_and_stderr() as out:
            timefun(_expensive_function)()
        msg = str(out)
        # print(out)
        assert len(out), msg
        assert out[0].startswith("@timefun:_expensive_function took"), msg
        time = float(out[0].split("took")[1].strip().split(" ")[0])
        assert time > 0


class TestTimeWith:
    def test_timing_with_context_manager(self):
        # prints something like:
        # fancy thing done with something took 0.582462072372 seconds
        # fancy thing done with something else took 1.75355315208 seconds
        # fancy thing finished took 1.7535982132 seconds
        with capture_stdout_and_stderr() as out:
            with TimeWith("fancy thing") as timer:
                _expensive_function()
                timer.checkpoint("done with something")
                _expensive_function()
                _expensive_function()
                timer.checkpoint("done with something else")

        msg = str(out)
        # print(out)
        assert len(out), msg
        out0 = out[0].split("\n")
        assert out0[0].startswith("fancy thing done with something took"), msg
        assert out0[1].startswith("fancy thing done with something else took"), msg
        assert out0[2].startswith("fancy thing finished took"), msg

    def test_direct_timing(self):
        # or directly
        with capture_stdout_and_stderr() as out:
            timer = TimeWith("fancy thing")
            _expensive_function()
            timer.checkpoint("done with something")

        msg = str(out)
        # print(out)
        assert len(out), msg
        assert out[0].startswith("fancy thing done with something took"), msg


class TestDoCProfile:
    def test_on_function(self):
        @do_cprofile
        def expensive_function():
            for x in _get_number():
                i = x**3
            return i

        with capture_stdout_and_stderr() as out:
            expensive_function()
        results = _extract_do_cprofile_results(out[0])
        print(results)
        msg = str(results)
        assert len(results), msg
        assert results[0][5].startswith("function calls in")
        assert results[0][0] > 50000
        assert results[1] == "ncalls  tottime  percall  cumtime  percall filename:lineno(function)"
        num_tests = 0
        for result in results[2:]:
            if result[5].endswith("(expensive_function)"):
                num_tests += 1
                for i in range(5):
                    assert result[i] > 0
            elif result[5].endswith("(_get_number)"):
                num_tests += 1
        if num_tests != 2:
            raise ValueError("Did not find _get_number or expensive_function")


def row_finder(rows: list[Any]) -> Callable[[str], tuple]:
    def find_row(text: str) -> tuple:
        row = next(
            (row for row in rows if isinstance(row, tuple) and row[5].strip() == text),
            None,
        )
        assert row is not None, f"Could not find row: {text!r}"
        return row

    return find_row


#  @pytest.mark.skip('Suspect this test fucks up coverage stats.')
@pytest.mark.skipif(LineProfiler is None, reason="LineProfiler is not installed.")
class TestDoProfile:
    def test_on_function_and_follow_function(self):
        @do_profile(follow=[_get_number])
        def expensive_function():
            for x in _get_number():
                i = x**3
            return i

        with capture_stdout_and_stderr() as out:
            expensive_function()
        results = _extract_do_profile_results(out[0])
        msg = str(results)

        assert len(results) > 0, msg
        assert results[0] == FIRST_LINE, msg

        find_row = row_finder(results)
        find_row("def _get_number():")
        find_row("@do_profile(follow=[_get_number])")

    def test_on_class_method_and_follow_function(self):
        class ExpensiveClass1:
            @do_profile(follow=[_get_number])
            def expensive_method1(self):
                for x in _get_number():
                    i = x ^ 6
                return i

        with capture_stdout_and_stderr() as out:
            ExpensiveClass1().expensive_method1()
        results = _extract_do_profile_results(out[0])
        print(results)
        msg = str(results)
        assert len(results) > 0, msg
        assert results[0] == FIRST_LINE, msg

        find_row = row_finder(results)
        find_row("def _get_number():")
        find_row("@do_profile(follow=[_get_number])")

    def test_on_class_method_and_follow_class_method(self):
        class ExpensiveClass2:
            n = 5000
            """You can not put class method _get_number2 directly into follow
            instead you must pass its name as a string:
            """

            @do_profile(follow=["_get_number2"])
            def expensive_method2(self):
                for x in self._get_number2():
                    i = x**4
                return i

            def _get_number2(self):
                yield from range(self.n)

        with capture_stdout_and_stderr() as out:
            ExpensiveClass2().expensive_method2()
        results = _extract_do_profile_results(out[0])
        print(results)
        msg = str(results)
        assert len(results) > 0, msg
        assert results[0] == FIRST_LINE, msg

        find_row = row_finder(results)
        find_row('@do_profile(follow=["_get_number2"])')
        assert find_row("def expensive_method2(self):")[1] == 0, msg
        find_row("def _get_number2(self):")

    def test_on_all_class_methods(self):
        class ExpensiveClass3:
            n = 5000
            n2 = 50
            """Profile all methods of ExpensiveClass3"""

            @do_profile(follow_all_methods=True)
            def expensive_method3(self):
                for x in self._get_number3():
                    for _ in self._get_number32():
                        i = x ^ 9
                return i

            def _get_number3(self):
                yield from range(self.n)

            def _get_number32(self):
                yield from range(self.n2)

        with capture_stdout_and_stderr() as out:
            ExpensiveClass3().expensive_method3()
        results = _extract_do_profile_results(out[0])
        msg = str(results)
        assert len(results) > 0, msg
        assert results[0] == FIRST_LINE

        find_row = row_finder(results)
        find_row("@do_profile(follow_all_methods=True)")
        row = find_row("def expensive_method3(self):")
        assert row[1] == 0
        assert row[2] == 0

        find_row("def _get_number3(self):")

    def test_on_all_class_methods_without_decorator(self):
        with capture_stdout_and_stderr() as out:
            cls = ExpensiveClass4()
            do_profile(follow=[cls._get_number4])(cls.expensive_method4)()
        results = _extract_do_profile_results(out[0])
        print(results)
        msg = str(results)
        assert len(results) > 0, msg
        assert results[0] == FIRST_LINE, msg
        find_row = row_finder(results)
        find_row("def expensive_method4(self):")
        row = find_row("for x in self._get_number4():")
        assert row[1] == 5001, msg
        assert row[2] > 10, msg

        find_row("def _get_number4(self):")
