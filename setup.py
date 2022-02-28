#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:49:13 2019

@author: ben
"""

from setuptools import setup, find_packages
from Cython.Build import cythonize
import logging
import sys
import numpy
import os

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
log = logging.getLogger()

# get long_description from README.md
with open("README.md", "r") as fh:
    long_description = fh.read()

# run cmd from the command line
def check_output(cmd):
    return subprocess.check_output(cmd).decode('utf')

scripts = [os.path.join('scripts',f) for f in os.listdir('scripts')]

setup(
    name='ATM_waveform',
    version='0.0.0.1',
    description='Utilities for fitting physically-based models to ATM data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/SmithB/ATM_waveform',
    author='Ben Smith',
    author_email='besmith@uw.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(),
    ext_modules = cythonize("ATM_waveform/corr_no_mean.pyx"),
    include_dirs=[numpy.get_include()],
    scripts=scripts
)
