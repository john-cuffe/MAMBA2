from setuptools import setup
from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    package_dir={'programs':''},
    ext_modules=cythonize(["febrl_methods.py","match_helpers.py","create_db_helpers.py","soundex.py", "model_generators.py", "score_functions.py"]),
)