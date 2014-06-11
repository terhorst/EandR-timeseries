#!/usr/bin/env python2.7

from distutils.core import setup, Extension
from Cython.Build import cythonize
import os
from glob import glob
import numpy as np

os.environ['CXX'] = "g++-4.8"
os.environ['CC'] = "gcc-4.8"

srcs = ["pyexp.pyx"]

setup(
    ext_modules = cythonize(Extension(
        "pyexp",
        srcs,
        include_dirs=[np.get_include()],
        language="c++"))
)

