import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
from numdifftools.testing import capture_stdout_and_stderr
from numdifftools.profiletools import do_profile, do_cprofile, timefun


def _extract_do_profile_results(txt, header_start='Line #'):
    results = []
    for line in txt.split('\n'):

        line = line.strip()
        if line.startswith(header_start):
            results.append(line)
            continue
        item, _, tail = line.partition(' ')
        try:
            line_no = int(item)
        except ValueError:
            continue
        try:
            vals = []
            for i in range(4):
                item, _, tail = tail.strip().partition(' ')
                vals.append(float(item))
            hits, time, perhit, percent_time = vals
        except ValueError:
            tail = ' '.join((item, tail))
            hits, time, perhit, percent_time = 0, 0, 0.0, 0.0

        results.append((line_no, hits, time, perhit, percent_time,
                        tail))

    return results


def _extract_do_cprofile_results(txt):
    """
     ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    5000001    0.326    0.000    0.326    0.000 testing.py:118(_get_number)
          1    0.858    0.858    1.184    1.184 testing.py:163(expensive_function2)
          1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
    """
    return _extract_do_profile_results(txt, header_start='ncalls')


def _get_number():
    for x in range(50000):
        yield x


class ExpensiveClass4(object):

    def expensive_method4(self):
        for x in self._get_number4():
            i = x **3
        return i

    def _get_number4(self):
        for x in range(5000):
            yield x

FIRST_LINE = 'Line #      Hits         Time  Per Hit   % Time  Line Contents'


class TestDoProfile(unittest.TestCase):

    def test_on_function_and_follow_function(self):
        @do_profile(follow=[_get_number])
        def expensive_function():
            for x in _get_number():
                i = x ** 3
            return i
        with capture_stdout_and_stderr() as out:
            _test0 = expensive_function()
        results = _extract_do_profile_results(out[0])
        self.assertEqual(results[0], FIRST_LINE)
        self.assertEqual(results[1][5].strip(), 'def _get_number():')
        self.assertEqual(results[2][5].strip(), 'for x in range(50000):')
        self.assertEqual(results[2][1], 50001)
        self.assertTrue(results[2][2] > 4000)
        self.assertEqual(results[4], FIRST_LINE)
        self.assertEqual(results[5][5].strip(),
                         '@do_profile(follow=[_get_number])')

    def test_on_class_method_and_follow_function(self):
        class ExpensiveClass1(object):

            @do_profile(follow=[_get_number])
            def expensive_method1(self):
                for x in _get_number():
                    i = x ^ 6
                return i

        with capture_stdout_and_stderr() as out:
            _test1 = ExpensiveClass1().expensive_method1()
        results = _extract_do_profile_results(out[0])
        print(results)
        self.assertEqual(results[0], FIRST_LINE)
        self.assertEqual(results[1][5].strip(),
                         'def _get_number():')
        self.assertEqual(results[2][5].strip(),
                         "for x in range(50000):")
        self.assertEqual(results[2][1], 50001)
        self.assertTrue(results[2][2] > 2900)
        self.assertEqual(results[4], FIRST_LINE)
        self.assertEqual(results[5][5].strip(),
                         '@do_profile(follow=[_get_number])')

    def test_on_class_method_and_follow_class_method(self):
        class ExpensiveClass2(object):
            """You can not put class method _get_number2 directly into follow
            instead you must pass its name as a string:
            """
            @do_profile(follow=['_get_number2'])
            def expensive_method2(self):
                for x in self._get_number2():
                    i = x ^ x ^ x ^ x
                return i

            def _get_number2(self):
                for x in xrange(5000):
                    yield x
        with capture_stdout_and_stderr() as out:
            _test2 = ExpensiveClass2().expensive_method2()
        results = _extract_do_profile_results(out[0])
        print(results)
        self.assertEqual(results[0], FIRST_LINE)
        self.assertEqual(results[1][5].strip(),
                         "@do_profile(follow=['_get_number2'])")
        self.assertEqual(results[2][5].strip(),
                         'def expensive_method2(self):')
        self.assertEqual(results[2][1], 0)
        self.assertEqual(results[2][2], 0)
        self.assertEqual(results[6], FIRST_LINE)
        self.assertEqual(results[7][5].strip(), 'def _get_number2(self):')

    def test_on_all_class_methods(self):
        class ExpensiveClass3(object):
            """Profile all methods of ExpensiveClass3"""
            @do_profile(follow_all_methods=True)
            def expensive_method3(self):
                for x in self._get_number3():
                    for y in self._get_number32():
                        i = x ^ 9
                return i

            def _get_number3(self):
                for x in range(5000):
                    yield x

            def _get_number32(self):
                for x in range(50):
                    yield x
        with capture_stdout_and_stderr() as out:
            _test3 = ExpensiveClass3().expensive_method3()
        results = _extract_do_profile_results(out[0])
        self.assertEqual(results[0], FIRST_LINE)
        self.assertEqual(results[1][5].strip(),
                         '@do_profile(follow_all_methods=True)')
        self.assertEqual(results[2][5].strip(),
                         'def expensive_method3(self):')
        self.assertEqual(results[2][1], 0)
        self.assertEqual(results[2][2], 0)
        self.assertEqual(results[7], FIRST_LINE)
        self.assertEqual(results[8][5].strip(), 'def _get_number3(self):')

    def test_on_all_class_methods_without_decorator(self):
        with capture_stdout_and_stderr() as out:
            cls = ExpensiveClass4()
            _test4 = do_profile(
                follow=[
                    cls._get_number4])(
                cls.expensive_method4)()
        results = _extract_do_profile_results(out[0])

        print(results)
        self.assertEqual(results[0], FIRST_LINE)
        self.assertEqual(results[1][5].strip(),
                         'def expensive_method4(self):')
        self.assertEqual(results[2][5].strip(),
                         'for x in self._get_number4():')
        self.assertEqual(results[2][1], 5001)
        self.assertTrue(results[2][2] > 10)
        self.assertEqual(results[5], FIRST_LINE)
        self.assertEqual(results[6][5].strip(), 'def _get_number4(self):')


class TestDoCProfile(unittest.TestCase):

    def test_on_function(self):
        @do_cprofile
        def expensive_function():
            for x in _get_number():
                i = x ** 3
            return i
        with capture_stdout_and_stderr() as out:
            _test0 = expensive_function()
        results = _extract_do_cprofile_results(out[0])
        print(results)
        self.assertTrue(results[0][5].startswith('function calls in'))
        self.assertTrue(results[0][0] > 50000)
        self.assertEqual(results[1],
                         'ncalls  tottime  percall  cumtime  percall filename:lineno(function)')
        for i in range(5):
            self.assertGreater(results[2][i], 0)
        self.assertEqual(
            results[2][5],
            'test_profiletools.py:192(expensive_function)')
        self.assertEqual(results[3][5], 'test_profiletools.py:47(_get_number)')


class TestTimeFun(unittest.TestCase):

    def test_on_function(self):
        @timefun
        def expensive_function():
            for x in _get_number():
                i = x ** 3
            return i
        with capture_stdout_and_stderr() as out:
            _test0 = expensive_function()
        print(out[0])
        self.assertTrue(out[0].startswith('@timefun:expensive_function took'))
        time = float(out[0].split('took')[1].strip().split(' ')[0])
        self.assertTrue(time > 0)
