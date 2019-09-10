# -*- coding: utf-8 -*-
from distutils.core import setup
from Cython.Build import cythonize

setup(
name='encode',
ext_modules=cythonize('encode.pyx'),)