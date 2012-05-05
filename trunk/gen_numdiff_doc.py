"""
Runs epydoc to make html documentation for Numdifftools
"""
import os
print('Generating html documentation for Numdifftools in folder html.')

os.system("epydoc.py --debug --html -v -o html --docformat reStructuredText --url http://code.google.com/p/numdifftools/ --name numdifftools --graph all numdifftools")