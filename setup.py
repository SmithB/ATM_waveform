#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:49:13 2019

@author: ben
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy
setup(
    ext_modules = cythonize("corr_no_mean_cython.pyx"),
    include_dirs=[numpy.get_include()]
)
