import os


def test_all():
    os.system('nosetests -w numdifftools/ --with-doctest')

if __name__ == '__main__':
    test_all()