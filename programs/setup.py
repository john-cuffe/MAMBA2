from setuptools import setup
from Cython.Build import cythonize

setup(
    package_dir={'programs':''},
    ext_modules=cythonize("febrl_methods.pyx"),
)