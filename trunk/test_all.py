import os


def test_all():
    os.system('nosetests -w numdifftools/ --with-doctest --doctest-options=+NORMALIZE_WHITESPACE')

if __name__ == '__main__':
    test_all()