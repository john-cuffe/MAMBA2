# -*- coding: utf-8 -*-
from distutils.core import setup

from Cython.Build import cythonize

setup(
name='febrl',
ext_modules=cythonize('mymath.pyx'),)