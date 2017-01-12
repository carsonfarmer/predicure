#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A sketch-based semi-Markov learner with approximate duration estimation.

Installation script for predicure.
"""

# Copyright © 2017 Carson J. Q. Farmer <carson@set.gl>
# All rights reserved. BSD-2 Licensed.

import sys
import os
import warnings

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

PACKAGE_NAME = "predicure"
DESCRIPTION = "predicure: A sketch-based semi-Markov model."

MAJOR = 0
MINOR = 1
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
QUALIFIER = ''
FULLVERSION = VERSION
if not ISRELEASED:
    FULLVERSION += '.dev'
    try:
        import subprocess
        try:
            pipe = subprocess.Popen(["git", "rev-parse", "--short", "HEAD"],
                                    stdout=subprocess.PIPE).stdout
        except OSError:
            # msysgit compatibility
            pipe = subprocess.Popen(
                ["git.cmd", "describe", "HEAD"],
                stdout=subprocess.PIPE).stdout
        rev = pipe.read().strip()
        # Makes distutils blow up on Python 2.7
        if sys.version_info[0] >= 3:
            rev = rev.decode('ascii')
        FULLVERSION = '%d.%d.%d.dev-%s' % (MAJOR, MINOR, MICRO, rev)
    except:
        warnings.warn("Couldn't get git revision")
else:
    FULLVERSION += QUALIFIER


setup(name=PACKAGE_NAME, version=FULLVERSION, description=DESCRIPTION,
      license='BSD-2-Clause', author='Carson J. Q. Farmer',
      author_email='carson@set.gl',
      keywords="streaming probability algorithm online Markov model",
      long_description=DESCRIPTION, packages=find_packages("."),
      install_requires=["streamhist"], zip_safe=True,
      setup_requires=["pytest-runner",], tests_require=["pytest",],
      classifiers=["Development Status :: 2 - Pre-Alpha",
                   "Environment :: Console",
                   "Intended Audience :: Science/Research",
                   "Intended Audience :: Developers",
                   "Intended Audience :: Information Technology",
                   "License :: OSI Approved :: BSD-2-Clause License",
                   "Natural Language :: English",
                   "Operating System :: OS Independent",
                   "Programming Language :: Python",
                   "Topic :: Scientific/Engineering :: Information Analysis",
                   "Topic :: System :: Distributed Computing",
                   ])
