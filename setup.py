#!/usr/bin/env python
# -*- coding: utf-8 -*-


import glob

from setuptools import setup, find_packages

# Versions
_version_major = 0
_version_minor = 1
_version_micro = 0
_version_extra = ''

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)
__version__ = '.'.join(map(str, _ver))

# PyPi Package Information
CLASSIFIERS = ["Development Status :: 1 - Planning",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Load long description
with open("README.md", "r") as fh:
    long_description = fh.read()

# Read the requirements
with open('requirements.txt') as f:
    required_dependencies = f.read().splitlines()
    external_dependencies = []
    for dependency in required_dependencies:
        if dependency[0:2] == '-e':
            repo_name = dependency.split('=')[-1]
            repo_url = dependency[3:]
            external_dependencies.append('{} @ {}'.format(repo_name, repo_url))
        else:
            external_dependencies.append(dependency)

opts = dict(name="MegaStitch",
            python_requires=">=3.10",
            version=__version__,
            packages=find_packages(),
            setup_requires=[],
            install_requires=external_dependencies,
            scripts=glob.glob("scripts/*.py"),
            include_package_data=True)

setup(**opts)